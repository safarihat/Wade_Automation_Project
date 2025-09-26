
import os
from django.conf import settings
import json
import time
import requests
import logging
import hashlib
import redis
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, RetryError

import math
MOCK_ARCGIS_DATA = os.environ.get('MOCK_ARCGIS_DATA', 'False') == 'True'

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

try:
    from pyproj import Transformer, CRS
except ImportError:
    Transformer = None
    CRS = None

logger = logging.getLogger(__name__)

# --- Redis Cache Setup ---
# Connect to Redis using the same URL as Celery for consistency.
try:
    redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
    logger.info("Successfully connected to Redis for caching.")
except Exception as e:
    logger.error(f"Could not connect to Redis for caching: {e}. Caching will be disabled.", exc_info=True)
    redis_client = None

def transform_coords(lon: float, lat: float, from_epsg: int, to_epsg: int) -> tuple[float, float]:
    """Transforms coordinates from one EPSG to another."""
    if not Transformer or not CRS:
        raise ImportError("pyproj is not installed. Please install it to use coordinate transformation features.")
    transformer = Transformer.from_crs(CRS(f"EPSG:{from_epsg}"), CRS(f"EPSG:{to_epsg}"), always_xy=True)
    x, y = transformer.transform(lon, lat)
    return x, y

def load_and_embed_documents():
    data_dir = os.path.join(settings.BASE_DIR, 'doc_generator', 'data', 'context')
    all_documents = []

    pdf_loader = DirectoryLoader(data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True)
    all_documents.extend(pdf_loader.load())

    txt_loader = DirectoryLoader(data_dir, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'}, show_progress=True)
    all_documents.extend(txt_loader.load())
    
    if not all_documents:
        logger.warning("No documents were loaded from 'doc_generator/data/context'.")
        return 0

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(all_documents)

    ids = [hashlib.md5(f"{doc.metadata['source']}-{doc.page_content}".encode()).hexdigest() for doc in texts]

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
    
    vector_store_path = os.path.join(settings.BASE_DIR, 'vector_store')
    vector_store = Chroma(persist_directory=vector_store_path, embedding_function=embeddings)
    vector_store.add_documents(documents=texts, ids=ids)
    vector_store.persist()
    
    return len(all_documents)

# =============================================================================
# EXTERNAL API QUERY HELPERS (REFACTORED)
# =============================================================================

def _query_koordinates_vector(layer_id: int, lon: float, lat: float, api_key: str, radius: int = 100, max_results: int = 10) -> list | None:
    """Queries a Koordinates vector layer. Returns a list of features or None on failure."""
    url = "https://koordinates.com/services/query/v1/vector.json"
    params = {'key': api_key, 'layer': layer_id, 'x': lon, 'y': lat, 'crs': 'epsg:4326', 'max_results': max_results, 'radius': radius, 'geometry': 'true', 'with_field_names': 'true'}
    
    cache_key = f"koordinates_vector:{layer_id}:{lon}:{lat}:{radius}:{max_results}"
    if redis_client:
        try:
            cached_result = redis_client.get(cache_key)
            if cached_result:
                logger.info(f"Cache HIT for Koordinates vector layer {layer_id}")
                return json.loads(cached_result)
        except redis.RedisError as e:
            logger.warning(f"Redis GET failed for Koordinates vector: {e}. Proceeding without cache.")


    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
        logger.info(f"Koordinates vector response for layer {layer_id}: {json.dumps(data)}")
        
        # Safely access and aggregate features from ALL layers in the response.
        all_features = []
        layers = data.get('vectorQuery', {}).get('layers')
        
        # Handle case where API returns an empty dictionary {} for out-of-extent queries
        if isinstance(layers, dict) and not layers:
            logger.warning(f"Koordinates vector query for layer {layer_id} at ({lon}, {lat}) returned an empty 'layers' object, likely outside extent.")

        if isinstance(layers, list) and layers:
            for layer in layers:
                if isinstance(layer, dict):
                    all_features.extend(layer.get('features', []))

        result = []
        if all_features:
            result = [{"properties": f.get("properties", {}), "geometry": f.get("geometry", {}), "distance": f.get("distance")} for f in all_features]
        
        if redis_client:
            try:
                redis_client.set(cache_key, json.dumps(result), ex=3600) # Cache for 1 hour
            except redis.RedisError as e:
                logger.warning(f"Redis SET failed for Koordinates vector: {e}.")
        
        logger.info(f"No Koordinates features found for layer {layer_id} at ({lon}, {lat}).")
        return [] # Return empty list for no features
    except requests.RequestException as e:
        logger.error(f"Koordinates vector query failed for layer {layer_id}: {e}", exc_info=True)
        return None # Return None on error

def _query_koordinates_raster(layer_ids: dict[str, int], lon: float, lat: float, api_key: str) -> dict | None:
    """
    Queries one or more Koordinates raster layers by their IDs.
    `layer_ids` is a dictionary mapping a name (e.g., 'elevation') to a layer ID (e.g., 51768).
    Returns a dictionary of the results (e.g., {'elevation': 100.5, 'slope': 12.3}) or None on failure.
    """
    url = "https://koordinates.com/services/query/v1/raster.json"
    
    # Create a mapping from layer ID back to the name for parsing the response
    id_to_name_map = {v: k for k, v in layer_ids.items()}
    layer_id_string = ",".join(map(str, layer_ids.values()))

    params = {'key': api_key, 'layer': layer_id_string, 'x': lon, 'y': lat, 'crs': 'epsg:4326'}

    cache_key = f"koordinates_raster:{layer_id_string}:{lon}:{lat}"
    if redis_client:
        try:
            cached_result = redis_client.get(cache_key)
            if cached_result:
                logger.info(f"Cache HIT for Koordinates raster layers {layer_id_string}")
                return json.loads(cached_result)
        except redis.RedisError as e:
            logger.warning(f"Redis GET failed for Koordinates raster: {e}. Proceeding without cache.")
    
    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
        logger.info(f"Koordinates raster response for layers {layer_id_string}: {json.dumps(data)}")
        
        # The rasterQuery response has a 'layers' dictionary keyed by layer ID.
        results = {}
        layers_dict = data.get('rasterQuery', {}).get('layers', {})
        if not isinstance(layers_dict, dict):
            logger.warning(f"Expected 'layers' to be a dict, but got {type(layers_dict)}. Response: {data}")
            return {}

        for layer_id_str, layer_data in layers_dict.items():
            layer_id = int(layer_id_str)
            result_name = id_to_name_map.get(layer_id)

            if not result_name:
                continue
            
            # Check for "outside-extent" status, which indicates no data is available.
            if layer_data.get('status') == 'outside-extent':
                logger.warning(f"Coordinates ({lon}, {lat}) are outside the extent of layer {layer_id} ({result_name}).")
                continue

            if not (isinstance(layer_data, dict) and 'bands' in layer_data and layer_data['bands']):
                continue

            # Extract the value from the first band.
            value = layer_data['bands'][0].get('value')
            if value is not None and value != 'NoData':
                try:
                    results[result_name] = float(value)
                except (ValueError, TypeError):
                    logger.warning(f"Could not parse raster value '{value}' for layer {layer_id}.")

        if redis_client:
            try:
                redis_client.set(cache_key, json.dumps(results), ex=3600) # Cache for 1 hour
            except redis.RedisError as e:
                logger.warning(f"Redis SET failed for Koordinates raster: {e}.")

        logger.info(f"Parsed raster results: {results}")
        return results
    except requests.RequestException as e:
        logger.error(f"Koordinates raster query failed for layers {layer_id_string}: {e}", exc_info=True)
        return None

def _calculate_slope_from_dem(lon: float, lat: float, api_key: str, dem_layer_id: int = 51768) -> float | None:
    """
    Calculates slope by querying a 3x3 grid from a DEM layer if a direct slope query fails.
    Returns slope in degrees or None.
    """
    logger.info(f"Performing fallback slope calculation using DEM layer {dem_layer_id}.")
    grid_spacing_deg = 0.0001  # Approx 10 meters

    # 1. Define the 3x3 grid of coordinates
    lons = [lon - grid_spacing_deg, lon, lon + grid_spacing_deg]
    lats = [lat - grid_spacing_deg, lat, lat + grid_spacing_deg]
    grid_coords_lon = [l for l in lons for _ in range(3)]
    grid_coords_lat = [l for _ in range(3) for l in lats]

    # 2. Query each of the 9 points individually, as Koordinates raster API doesn't support multi-point queries.
    elevations = []
    for i in range(9):
        point_lon, point_lat = grid_coords_lon[i], grid_coords_lat[i]
        result = _query_koordinates_raster({'elevation': dem_layer_id}, point_lon, point_lat, api_key)
        if result and 'elevation' in result:
            elevations.append(result['elevation'])
        else:
            elevations.append(None)

    # 3. Check if we have all 9 points before calculating
    if None in elevations:
        logger.warning(f"Failed to get all 9 elevation points for slope calculation at ({lon}, {lat}). Got: {elevations}")
        return None

    # 4. Calculate slope using finite differences
    try:
        # Elevations grid: z[row][col]
        z = [elevations[i:i+3] for i in range(0, 9, 3)]

        # Calculate distances in meters
        earth_radius = 6371000
        dlat_rad = math.radians(grid_spacing_deg)
        dlon_rad = math.radians(grid_spacing_deg)
        dy = earth_radius * dlat_rad
        dx = earth_radius * dlon_rad * math.cos(math.radians(lat))

        # Central difference method for derivatives
        dz_dx = (z[1][2] - z[1][0]) / (2 * dx)
        dz_dy = (z[2][1] - z[0][1]) / (2 * dy)

        slope_rad = math.atan(math.sqrt(dz_dx**2 + dz_dy**2))
        slope_deg = math.degrees(slope_rad)
        logger.info(f"Calculated slope from DEM: {slope_deg:.2f} degrees.")
        return slope_deg
    except (ZeroDivisionError, IndexError, TypeError) as e:
        logger.error(f"Error during slope calculation math for ({lon}, {lat}): {e}", exc_info=True)
        return None

@retry(
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(5),
    # Only retry on 5xx errors or network issues, not 4xx
    retry=retry_if_exception_type(requests.exceptions.ConnectionError) |
          retry_if_exception_type(requests.exceptions.Timeout) |
          retry_if_exception_type(requests.exceptions.RequestException), # Catch all requests exceptions for retry
          # Exclude HTTPError specifically for 4xx, handled below
    reraise=True # Re-raise the exception after retries are exhausted
)
def _make_arcgis_request(url: str, params: dict) -> dict | None:
    """
    Makes a request to an ArcGIS service with a specified timeout.
    Handles 404s gracefully and retries on transient errors (5xx, network issues).
    """
    if MOCK_ARCGIS_DATA:
        logger.info(f"MOCK_ARCGIS_DATA is True. Returning mock data for {url}")
        if "FeatureServer" in url:
            return {"features": [{"attributes": {"SoilType": "MockLoam", "DominantSoilCode": "ML"}}]}
        elif "ImageServer" in url:
            return {"value": "10.5"} # Mock slope value
        return {}

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status() # Raise HTTPError for bad status codes (4xx or 5xx)
        return response.json()
    except requests.exceptions.HTTPError as e:
        if 400 <= e.response.status_code < 500:
            logger.warning(f"ArcGIS client error ({e.response.status_code}) for URL: {url} with params: {params}. Error: {e.response.text}")
            return None # Return None for client errors (4xx) to allow graceful handling upstream
        else: # 5xx server errors
            logger.error(f"ArcGIS server error ({e.response.status_code}) for {url}: {e}", exc_info=True)
            raise # Re-raise to trigger tenacity retries for server errors
    except requests.exceptions.RequestException as e:
        logger.error(f"ArcGIS request failed for {url}: {e}", exc_info=True)
        raise # Re-raise to trigger tenacity retries

def _query_arcgis_vector(service_url: str, lon: float, lat: float) -> list | None:
    """
    Queries an ArcGIS vector layer with caching and robust retries.
    Returns a list of feature attributes or None on failure.
    """
    # The service_url parameter will now be the new FeatureServer Layer 5 URL directly.
    # No need for domain fix here as the new URL is correct.

    cache_key = f"arcgis_vector:{hashlib.md5(f'{service_url}{lon}{lat}'.encode()).hexdigest()}"
    if redis_client:
        try:
            cached_result = redis_client.get(cache_key)
            if cached_result:
                logger.info(f"Cache HIT for ArcGIS vector: {service_url}")
                return json.loads(cached_result)
        except redis.RedisError as e:
            logger.warning(f"Redis GET failed: {e}. Proceeding without cache.")

    try:
        x, y = transform_coords(lon, lat, 4326, 2193) # Transform to NZTM (EPSG:2193)
        url = service_url # The service_url is already the full query endpoint

        params = {
            'geometry': f'{x},{y}',
            'geometryType': 'esriGeometryPoint',
            'inSR': 2193,
            'spatialRel': 'esriSpatialRelIntersects',
            'outFields': '*',
            'returnGeometry': 'false',
            'f': 'json'
        }

        data = _make_arcgis_request(url, params)

        if data is None: # Handle 4xx from _make_arcgis_request
            logger.info(f"ArcGIS vector query returned no data (e.g., 4xx) for {service_url}.")
            return [] # Return empty list as fallback

        logger.info(f"ArcGIS vector response from {service_url}: {json.dumps(data)}")

        if 'features' in data and data['features']:
            result = [f['attributes'] for f in data['features']]
            if redis_client:
                try:
                    redis_client.set(cache_key, json.dumps(result), ex=3600) # Cache for 1 hour
                except redis.RedisError as e:
                    logger.warning(f"Redis SET failed: {e}.")
            return result

        logger.info(f"No vector features found at this location from {service_url}.")
        return [] # Return empty list for no features

    except (RetryError, requests.RequestException) as e:
        logger.error(f"ArcGIS vector query failed after retries for {service_url}: {e}", exc_info=True)
        return [] # Return empty list on persistent failure
    except Exception as e:
        logger.error(f"An unexpected error occurred in _query_arcgis_vector: {e}", exc_info=True)
        return []
