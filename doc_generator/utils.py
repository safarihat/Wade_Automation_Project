
import os
from django.conf import settings
import json
import time
import requests
import logging
import hashlib
import redis
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, RetryError

MOCK_ARCGIS_DATA = os.environ.get('MOCK_ARCGIS_DATA', 'False') == 'True'

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

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
    
    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
        logger.info(f"Koordinates vector response for layer {layer_id}: {json.dumps(data)}")
        
        # Safely access and aggregate features from ALL layers in the response.
        all_features = []
        layers = data.get('vectorQuery', {}).get('layers')
        if isinstance(layers, list) and layers:
            for layer in layers:
                if isinstance(layer, dict):
                    all_features.extend(layer.get('features', []))

        if all_features:
            return [{"properties": f.get("properties", {}), "geometry": f.get("geometry", {}), "distance": f.get("distance")} for f in all_features]
        
        logger.info(f"No Koordinates features found for layer {layer_id} at ({lon}, {lat}).")
        return [] # Return empty list for no features
    except requests.RequestException as e:
        logger.error(f"Koordinates vector query failed for layer {layer_id}: {e}", exc_info=True)
        return None # Return None on error

def _query_koordinates_raster(layer_id: int, lon: float, lat: float, api_key: str) -> dict | None:
    """Queries a Koordinates raster layer. Returns the layer data or None on failure."""
    url = "https://koordinates.com/services/query/v1/raster.json"
    params = {'key': api_key, 'layer': layer_id, 'x': lon, 'y': lat, 'crs': 'epsg:4326'}
    
    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
        logger.info(f"Koordinates raster response for layer {layer_id}: {json.dumps(data)}")
        
        # Find the first valid layer with bands in the response list.
        layers = data.get('rasterQuery', {}).get('layers', [])
        for layer in layers:
            if isinstance(layer, dict) and 'bands' in layer:
                return layer
            
        logger.info(f"No Koordinates raster value found for layer {layer_id} at ({lon}, {lat}).")
        return {} # Return empty dict for no value
    except requests.RequestException as e:
        logger.error(f"Koordinates raster query failed for layer {layer_id}: {e}", exc_info=True)
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

def _query_arcgis_raster(service_url: str, lon: float, lat: float) -> dict | None:
    """
    Queries an ArcGIS raster layer with caching and robust retries.
    Returns a dictionary with the raster value or None on failure.
    """
    # The service_url parameter will now be the new ImageServer URL directly.
    # No need for domain fix here as the new URL is correct.

    cache_key = f"arcgis_raster:{hashlib.md5(f'{service_url}{lon}{lat}'.encode()).hexdigest()}"
    if redis_client:
        try:
            cached_result = redis_client.get(cache_key)
            if cached_result:
                logger.info(f"Cache HIT for ArcGIS raster: {service_url}")
                return json.loads(cached_result)
        except redis.RedisError as e:
            logger.warning(f"Redis GET failed: {e}. Proceeding without cache.")

    try:
        # New URL for raster: https://elevation.arcgis.com/arcgis/rest/services/WorldElevation/Terrain/ImageServer/identify
        # The service_url parameter will be this full URL.
        url = service_url # The service_url is already the full identify endpoint

        params = {
            'geometry': f'{lon},{lat}', # Use WGS84 lon,lat directly
            'geometryType': 'esriGeometryPoint',
            'sr': 4326, # Spatial reference for input geometry
            'tolerance': 2,
            'returnGeometry': 'false',
            'f': 'json'
        }

        data = _make_arcgis_request(url, params)

        if data is None: # Handle 4xx from _make_arcgis_request
            logger.info(f"ArcGIS raster query returned no data (e.g., 4xx) for {service_url}.")
            return {"value": None} # Return dict with None value as fallback

        logger.info(f"ArcGIS raster response from {service_url}: {json.dumps(data)}")

        result_value = None
        if 'value' in data and data['value'] is not None and data['value'] != 'NoData':
            result_value = data['value']
        elif 'results' in data and data['results']:
            props = data['results'][0].get('attributes', {})
            pixel_value = props.get('Pixel Value', props.get('pixel_value'))
            if pixel_value is not None and pixel_value != 'NoData':
                result_value = pixel_value

        if result_value is not None:
            result = {'value': result_value}
            if redis_client:
                try:
                    redis_client.set(cache_key, json.dumps(result), ex=3600) # Cache for 1 hour
                except redis.RedisError as e:
                    logger.warning(f"Redis SET failed: {e}.")
            return result

        logger.info(f"No valid raster value found at this location from {service_url}.")
        return {"value": None} # Return dict with None value for no value

    except (RetryError, requests.RequestException) as e:
        logger.error(f"ArcGIS raster query failed after retries for {service_url}: {e}", exc_info=True)
        return {"value": None} # Return dict with None value on persistent failure
    except Exception as e:
        logger.error(f"An unexpected error occurred in _query_arcgis_raster: {e}", exc_info=True)
        return {"value": None}

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
