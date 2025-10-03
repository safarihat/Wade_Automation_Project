
import os
import re
import hashlib
import logging
from typing import List, Dict, Any
import json
import time
import requests
import redis
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, RetryError
from arcgis2geojson import arcgis2geojson
import geojson
import math

from django.conf import settings

try:
    from pyproj import Transformer, CRS
except ImportError:
    Transformer = None
    CRS = None

logger = logging.getLogger(__name__)

# --- Redis Cache Setup ---
try:
    redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
    logger.info("Successfully connected to Redis for caching.")
except Exception as e:
    logger.error(f"Could not connect to Redis for caching: {e}. Caching will be disabled.", exc_info=True)
    redis_client = None

MOCK_ARCGIS_DATA = os.environ.get('MOCK_ARCGIS_DATA', 'False') == 'True'


def transform_coords(lon: float, lat: float, from_epsg: int, to_epsg: int) -> tuple[float, float]:
    """Transforms coordinates from one EPSG to another."""
    if not Transformer or not CRS:
        raise ImportError("pyproj is not installed. Please install it to use coordinate transformation features.")
    transformer = Transformer.from_crs(CRS(f"EPSG:{from_epsg}"), CRS(f"EPSG:{to_epsg}"), always_xy=True)
    x, y = transformer.transform(lon, lat)
    return x, y

@retry(
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(requests.exceptions.ConnectionError) |
          retry_if_exception_type(requests.exceptions.Timeout) |
          retry_if_exception_type(requests.exceptions.RequestException),
    reraise=True
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
    except requests.RequestException as e:
        logger.error(f"ArcGIS request failed for {url}: {e}", exc_info=True)
        raise # Re-raise to trigger tenacity retries

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
        
        all_features = []
        layers = data.get('vectorQuery', {}).get('layers')

        if isinstance(layers, dict):
            if not layers:
                logger.warning(f"Koordinates vector query for layer {layer_id} at ({lon}, {lat}) returned an empty 'layers' object, likely outside extent.")
            else:
                for layer_data in layers.values():
                    if isinstance(layer_data, dict):
                        all_features.extend(layer_data.get('features', []))

        result = []
        if all_features:
            result = [{
                "properties": f.get("properties", {}),
                "geometry": f.get("geometry", {}),
                "distance": f.get("distance")
            } for f in all_features]
        
        if redis_client:
            try:
                redis_client.set(cache_key, json.dumps(result), ex=3600) # Cache for 1 hour
            except redis.RedisError as e:
                logger.warning(f"Redis SET failed for Koordinates vector: {e}.")
        
        return result
    except requests.RequestException as e:
        logger.error(f"Koordinates vector query failed for layer {layer_id}: {e}", exc_info=True)
        return None

def _query_koordinates_raster(layer_ids: dict[str, int] | int, lon: float, lat: float, api_key: str) -> dict | None:
    if isinstance(layer_ids, int):
        layer_ids = {'value': layer_ids}

    url = "https://koordinates.com/services/query/v1/raster.json"
    
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
        
        results = {}
        layers_dict = data.get('rasterQuery', {}).get('layers', {})
        if not isinstance(layers_dict, dict):
            logger.warning(f"Expected 'layers' to be a dict, but got {type(layers_dict)}. Response: {data}")
            return {}

        for layer_id_str, layer_data in layers_dict.items():
            layer_id = int(layer_id_str)
            result_name = id_to_name_map.get(layer_id)

            if not result_name or layer_data.get('status') == 'outside-extent':
                continue

            if not (isinstance(layer_data, dict) and 'bands' in layer_data and layer_data['bands']):
                continue

            value = layer_data['bands'][0].get('value')
            if value is not None and value != 'NoData':
                try:
                    results[result_name] = float(value)
                except (ValueError, TypeError):
                    logger.warning(f"Could not parse raster value '{value}' for layer {layer_id}.")

        if redis_client:
            try:
                redis_client.set(cache_key, json.dumps(results), ex=3600)
            except redis.RedisError as e:
                logger.warning(f"Redis SET failed for Koordinates raster: {e}.")

        return results
    except requests.RequestException as e:
        logger.error(f"Koordinates raster query failed for layers {layer_id_string}: {e}", exc_info=True)
        return None

def _calculate_slope_from_dem(lon: float, lat: float, api_key: str, dem_layer_id: int = 51768) -> float | None:
    logger.info(f"Performing fallback slope calculation using DEM layer {dem_layer_id}.")
    grid_spacing_deg = 0.0001

    lons = [lon - grid_spacing_deg, lon, lon + grid_spacing_deg]
    lats = [lat - grid_spacing_deg, lat, lat + grid_spacing_deg]
    grid_coords_lon = [l for l in lons for _ in range(3)]
    grid_coords_lat = [l for _ in range(3) for l in lats]

    elevations = []
    for i in range(9):
        point_lon, point_lat = grid_coords_lon[i], grid_coords_lat[i]
        result = _query_koordinates_raster({'elevation': dem_layer_id}, point_lon, point_lat, api_key)
        if result and 'elevation' in result:
            elevations.append(result['elevation'])
        else:
            elevations.append(None)

    if None in elevations:
        logger.warning(f"Failed to get all 9 elevation points for slope calculation at ({lon}, {lat}). Got: {elevations}")
        return None

    try:
        z = [elevations[i:i+3] for i in range(0, 9, 3)]
        earth_radius = 6371000
        dlat_rad = math.radians(grid_spacing_deg)
        dlon_rad = math.radians(grid_spacing_deg)
        dy = earth_radius * dlat_rad
        dx = earth_radius * dlon_rad * math.cos(math.radians(lat))
        dz_dx = (z[1][2] - z[1][0]) / (2 * dx)
        dz_dy = (z[2][1] - z[0][1]) / (2 * dy)
        slope_rad = math.atan(math.sqrt(dz_dx**2 + dz_dy**2))
        return math.degrees(slope_rad)
    except (ZeroDivisionError, IndexError, TypeError) as e:
        logger.error(f"Error during slope calculation math for ({lon}, {lat}): {e}", exc_info=True)
        return None

def _query_arcgis_vector(service_url: str, lon: float, lat: float) -> list | None:
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
        x, y = transform_coords(lon, lat, 4326, 2193)
        params = {
            'geometry': f'{x},{y}', 'geometryType': 'esriGeometryPoint', 'inSR': 2193,
            'spatialRel': 'esriSpatialRelIntersects', 'outSR': 4326, 'outFields': '*',
            'returnGeometry': 'true', 'f': 'json'
        }
        data = _make_arcgis_request(service_url, params)

        if data is None or 'features' not in data or not data['features']:
            return []

        valid_features = []
        for esri_feature in data['features']:
            try:
                converted_feature = arcgis2geojson(esri_feature)
                validated_feature = geojson.Feature(
                    geometry=converted_feature['geometry'],
                    properties=converted_feature['properties']
                )
                if validated_feature.is_valid:
                    valid_features.append(validated_feature)
            except Exception:
                logger.error(f"Critical error converting/validating feature: {esri_feature}", exc_info=True)

        if redis_client and valid_features:
            try:
                redis_client.set(cache_key, json.dumps(valid_features), ex=3600)
            except redis.RedisError as e:
                logger.warning(f"Redis SET failed for validated features: {e}.")

        return valid_features
    except (RetryError, requests.RequestException) as e:
        logger.error(f"ArcGIS vector query failed after retries for {service_url}: {e}", exc_info=True)
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred in _query_arcgis_vector: {e}", exc_info=True)
        return []

def _query_arcgis_raster(service_url: str, lon: float, lat: float) -> dict | None:
    if not service_url.endswith('/'):
        service_url += '/'
    identify_url = f"{service_url}identify"

    cache_key = f"arcgis_raster:{hashlib.md5(f'{identify_url}{lon}{lat}'.encode()).hexdigest()}"
    if redis_client:
        try:
            cached_result = redis_client.get(cache_key)
            if cached_result:
                return json.loads(cached_result)
        except redis.RedisError as e:
            logger.warning(f"Redis GET failed for ArcGIS raster: {e}. Proceeding without cache.")

    try:
        params = {
            'geometry': f'{lon},{lat}', 'geometryType': 'esriGeometryPoint', 'inSR': 4326,
            'returnGeometry': 'false', 'f': 'json'
        }
        data = _make_arcgis_request(identify_url, params)

        if data is None:
            return None

        pixel_value = data.get('value')
        if pixel_value is None or str(pixel_value) == 'NoData':
            return None

        result = {'value': pixel_value}
        if redis_client:
            redis_client.set(cache_key, json.dumps(result), ex=3600)
        return result
    except Exception as e:
        logger.error(f"An unexpected error occurred in _query_arcgis_raster: {e}", exc_info=True)
        return None
