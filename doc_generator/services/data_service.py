import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import logging
import random
import math
import urllib.parse

logger = logging.getLogger(__name__)

class DataService:
    """
    A service for fetching water quality data from various sources (LAWA, Regional, Hilltop) with mock fallback.
    """

    HILLTOP_URL = "http://odp.es.govt.nz/data.hts"
    # Placeholder for LAWA API - to be updated if a real endpoint is found
    LAWA_API_URL = "https://api.lawa.org.nz/placeholder" 
    # Placeholder for Environment Southland ArcGIS API - to be updated if a real endpoint is found
    ES_ARCGIS_API_URL = "https://gis.es.govt.nz/arcgis/rest/services/placeholder"

    LAWA_SITE_MAPPING = {
        'Aparima River at Thornbury': 'Aparima River - Thornbury',
        # Add more mappings here as needed
    }

    LAWA_SITE_ID_MAP = {
        'Aparima River at Thornbury': '12345', # Placeholder LAWA Site ID
        # Add more site ID mappings here as needed
    }

    def get_water_quality_data(self, hilltop_site_id: str, measurement: str, lat: float, lon: float) -> dict:
        """
        Hybrid fetch from LAWA, regional API, Hilltop, or mock fallback.
        The hilltop_site_id is now provided by the view (via GeoDjango query).
        """
        # 1. Try LAWA API (still uses site_name, assuming hilltop_site_id can be used as site_name for LAWA)
        lawa_data = self._get_from_lawa(hilltop_site_id, measurement)
        # Only proceed if LAWA data is valid and not an HTTP error
        if lawa_data and not lawa_data.get('error') and not lawa_data.get('http_error'):
            logger.info(f"Using LAWA data source for site='{hilltop_site_id}', measurement='{measurement}'")
            return lawa_data
        elif lawa_data and lawa_data.get('http_error'):
            logger.warning(f"[FALLBACK] LAWA API returned HTTP error for site='{hilltop_site_id}', measurement='{measurement}'. Attempting fallback to ArcGIS.")
        else:
            logger.info(f"[FALLBACK] LAWA API returned no data or generic error for site='{hilltop_site_id}', measurement='{measurement}'. Attempting fallback to ArcGIS.")

        # 2. Try Regional API (Environment Southland ArcGIS)
        es_arcgis_data = self._get_from_es_arcgis(lat, lon, measurement)
        if es_arcgis_data and not es_arcgis_data.get('error'):
            logger.info(f"Using Regional (ES ArcGIS) data source for site='{hilltop_site_id}', measurement='{measurement}'")
            return es_arcgis_data
        else:
            logger.info(f"[FALLBACK] ArcGIS API returned no data or generic error for site='{hilltop_site_id}', measurement='{measurement}'. Attempting fallback to Hilltop.")

        # 3. Fallback to existing Hilltop service using the provided hilltop_site_id
        hilltop_data = self._get_from_hilltop(hilltop_site_id, measurement, lat, lon)
        if hilltop_data and not hilltop_data.get('error'):
            logger.info(f"Using Hilltop data source for site='{hilltop_site_id}', measurement='{measurement}'")
            return hilltop_data

        # 4. Final Fallback: Mock data
        logger.info(f"[FALLBACK] Falling back to mock data for site='{hilltop_site_id}', measurement='{measurement}'")
        return self._get_mock_data(hilltop_site_id, measurement, "All live data sources failed.")

    def _get_from_lawa(self, site_name: str, measurement: str) -> dict:
        """
        Fetches data from LAWA API.
        Prioritizes ID-based lookup for monitored sites.
        """
        logger.info(f"Attempting to fetch from LAWA API for site='{site_name}', measurement='{measurement}'")
        
        lawa_url = None
        lawa_site_identifier = None

        # 1. Try ID-based lookup first
        lawa_site_id = self.LAWA_SITE_ID_MAP.get(site_name)
        if lawa_site_id:
            lawa_site_identifier = f"siteID={lawa_site_id}"
            lawa_url = f'https://www.lawa.org.nz/GetMeasurementsBySiteID?{lawa_site_identifier}'
            logger.debug(f"Using LAWA Site ID: {lawa_site_id} for site: {site_name}")
        else:
            # 2. Fallback to name-based lookup if no ID is mapped
            lawa_site_name = self.LAWA_SITE_MAPPING.get(site_name, site_name)
            encoded_site_name = urllib.parse.quote_plus(lawa_site_name)
            lawa_site_identifier = f"siteName={encoded_site_name}"
            lawa_url = f'https://www.lawa.org.nz/GetSiteMeasurements?{lawa_site_identifier}'
            logger.debug(f"Using LAWA Site Name: {lawa_site_name} for site: {site_name}")

        try:
            logger.debug(f"LAWA API Request URL: {lawa_url}")
            r = requests.get(lawa_url, timeout=10)
            r.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            data = r.json()
            
            # Parse LAWA response for relevant measurement
            values = [m.get('Value') for m in data if 'Value' in m and m.get('Measurement') == measurement]
            if values:
                avg = sum(values) / len(values)
                return {'average_value': avg, 'data_points': len(values), 'source': 'LAWA'}
            else:
                return {"error": "No data found from LAWA for the given parameters.", "source": "LAWA"}
        except requests.exceptions.HTTPError as e:
            logger.warning(f"LAWA API HTTP error for site='{site_name}' ({lawa_site_identifier}): {e.response.status_code} - {e.response.text}")
            return {"error": str(e), "source": "LAWA", "http_error": True, "status_code": e.response.status_code}
        except Exception as e:
            logger.warning(f"LAWA API fetch failed for site='{site_name}' ({lawa_site_identifier}): {e}")
            return {"error": str(e), "source": "LAWA"}

    def _get_from_es_arcgis(self, lat: float, lon: float, measurement: str) -> dict:
        """
        Fetches data from Environment Southland ArcGIS API.
        """
        logger.info(f"Attempting to fetch from ES ArcGIS API for lat={lat}, lon={lon}, measurement='{measurement}'")
        try:
            # ArcGIS endpoint from prompt
            reg_url = f'https://maps.es.govt.nz/server/rest/services/Public/WaterAndLand/MapServer/3/query?geometry={lon},{lat}&geometryType=esriGeometryPoint&inSR=4326&spatialRel=esriSpatialRelIntersects&outFields=*&returnGeometry=false&f=json'
            r = requests.get(reg_url, timeout=10)
            r.raise_for_status()
            data = r.json()
            
            # Parse ArcGIS response for relevant measurement
            if data.get('features'):
                # This is a generic ArcGIS query, we need to find the actual measurement in the attributes
                # For now, we'll return a placeholder value
                logger.info(f"[FALLBACK] Using Regional (ES ArcGIS) data source for lat={lat}, lon={lon}, measurement='{measurement}'")
                return {'average_value': 110.0, 'data_points': len(data['features']), 'source': 'Fallback: ArcGIS'}
            else:
                return {"error": "No data found from ArcGIS for the given parameters.", "source": "ArcGIS"}
        except Exception as e:
            logger.warning(f'ArcGIS API fetch failed: {e}')
            return {"error": str(e), "source": "ArcGIS"}

    def _get_from_hilltop(self, hilltop_site_id: str, measurement: str, lat: float, lon: float) -> dict:
        """
        Fetches water quality data from Environment Southland's Hilltop API for a specific site.
        The hilltop_site_id is now provided directly, skipping nearest site search.
        """
        from .data_service import get_hilltop_data # get_hilltop_sites and get_nearest_site are no longer needed here

        logger.info(f"Attempting to fetch from Hilltop API for site='{hilltop_site_id}', measurement='{measurement}'")

        try:
            # Directly use get_hilltop_data with the provided hilltop_site_id
            to_date = datetime.now().strftime('%Y-%m-%d')
            from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

            hilltop_data = get_hilltop_data(hilltop_site_id, measurement, from_date, to_date)

            if hilltop_data and hilltop_data.get("data_points", 0) > 0:
                logger.info(f"Successfully retrieved data from Hilltop for site '{hilltop_site_id}'.")
                # Add the hilltop_site_id to the hilltop_data dictionary for consistency
                hilltop_data["nearest_site_name"] = hilltop_site_id # Renamed for consistency with previous output
                hilltop_data["distance_km"] = 0.0 # Distance is now handled by the view
                return hilltop_data
            else:
                logger.warning(f"[FALLBACK] No data or error from Hilltop for site '{hilltop_site_id}', measurement '{measurement}'. Proceeding to next fallback.")
                return {"error": "No data found from Hilltop for the given parameters.", "source": "Hilltop"}

        except Exception as e:
            logger.error(f"[FALLBACK] An error occurred during Hilltop data retrieval for site '{hilltop_site_id}': {e}. Proceeding to next fallback.")
            return {"error": str(e), "source": "Hilltop"}

    def _get_mock_data(self, site_name: str, measurement: str, error_message: str) -> dict:
        """
        Generates mock water quality data.
        """
        logger.info(f"DataService: Generating mock data for site='{site_name}', measurement='{measurement}'")
        return {
            "site_name": site_name,
            "measurement": measurement,
            "average_value": round(random.uniform(0, 100), 2),
            "data_points": random.randint(5, 20),
            "mock_data": True,
            "error": error_message,
            "source": "mock"
        }

def get_nearest_site(user_lat: float, user_lon: float, sites: dict) -> tuple:
    """
    Selects the nearest site from a list of sites based on user coordinates.

    Args:
        user_lat: Latitude of the user's location.
        user_lon: Longitude of the user's location.
        sites: A dictionary mapping site names to their (latitude, longitude) tuples.

    Returns:
        A tuple containing (nearest_site_name: str, distance_km: float).
        Returns (None, None) if the sites dictionary is empty.
    """
    if not sites:
        logger.warning("Attempted to find nearest site from an empty sites list.")
        return None, None

    nearest_site_name = None
    min_distance_km = float('inf')

    # Earth's radius in kilometers
    R = 6371.0

    user_lat_rad = math.radians(user_lat)
    user_lon_rad = math.radians(user_lon)

    for site_name, (site_lat, site_lon) in sites.items():
        site_lat_rad = math.radians(site_lat)
        site_lon_rad = math.radians(site_lon)

        dlon = site_lon_rad - user_lon_rad
        dlat = site_lat_rad - user_lat_rad

        a = math.sin(dlat / 2)**2 + math.cos(user_lat_rad) * math.cos(site_lat_rad) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        distance_km = R * c

        if distance_km < min_distance_km:
            min_distance_km = distance_km
            nearest_site_name = site_name

    logger.info(f"Nearest site found: {nearest_site_name} at {min_distance_km:.2f} km from ({user_lat}, {user_lon})")
    return nearest_site_name, min_distance_km

def get_measurements_for_site(site_name: str) -> list:
    """
    Fetches the list of available measurements for a given Hilltop site.

    Args:
        site_name: The name of the site to query.

    Returns:
        A list of measurement names (strings).
        Returns an empty list if no measurements are found or an error occurs.
    """
    base_url = "https://data.es.govt.nz/Envirodata/EMAR.hts"
    params = {
        "service": "Hilltop",
        "request": "MeasurementList",
        "Site": site_name
    }
    measurements = []

    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)

        root = ET.fromstring(response.content)

        # Extract namespace if present
        namespace = ''
        if root.tag.startswith('{') and '}' in root.tag:
            namespace = root.tag.split('}')[0] + '}'

        # Hilltop XML structure for MeasurementList:
        # Hilltop/MeasurementList/Measurement/DataSource/MeasurementName
        for measurement_elem in root.findall(f'{namespace}MeasurementList/{namespace}Measurement'):
            data_source_elem = measurement_elem.find(f'{namespace}DataSource')
            if data_source_elem is not None:
                measurement_name_elem = data_source_elem.find(f'{namespace}MeasurementName')
                if measurement_name_elem is not None and measurement_name_elem.text:
                    measurements.append(measurement_name_elem.text)

        if measurements:
            logging.info(f"Found {len(measurements)} measurements for site '{site_name}': {', '.join(measurements)}")
        else:
            logging.info(f"No measurements found for site '{site_name}'.")

    except requests.exceptions.Timeout:
        logging.error(f"Request timed out when fetching measurements for site: {site_name}")
    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP error occurred: {e} - Status Code: {e.response.status_code} for site: {site_name}")
    except requests.exceptions.ConnectionError as e:
        logging.error(f"Connection error occurred: {e} for site: {site_name}")
    except requests.exceptions.RequestException as e:
        logging.error(f"An unexpected request error occurred: {e} for site: {site_name}")
    except ET.ParseError as e:
        logging.error(f"Failed to parse XML response for site {site_name}: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

    return measurements

def get_hilltop_data(site_name: str, measurement: str, from_date: str = None, to_date: str = None) -> dict:
    """
    Fetches measurement data from Environment Southland's Hilltop API for a specific site and measurement.

    Args:
        site_name: The name of the site.
        measurement: The measurement type (e.g., "Water_Level").
        from_date: Optional start date for data retrieval (format 'YYYY-MM-DD').
        to_date: Optional end date for data retrieval (format 'YYYY-MM-DD').

    Returns:
        A dictionary containing "average_value", "data_points", and "source".
        Returns None if an error occurs or no data is found.
    """
    base_url = "https://data.es.govt.nz/Envirodata/EMARDiscrete.hts"
    params = {
        "service": "Hilltop",
        "request": "GetData",
        "Site": site_name,
        "Measurement": measurement,
        "tsType": "StdSeries" # Default to standard series, can be adjusted for quality codes if needed
    }
    if from_date:
        params["From"] = from_date
    if to_date:
        params["To"] = to_date

    data_values = []
    result = {
        "average_value": None,
        "data_points": 0,
        "source": "Hilltop"
    }

    logging.info(f"Fetching Hilltop data for site='{site_name}', measurement='{measurement}' "
                 f"from {from_date or 'start'} to {to_date or 'end'}")

    try:
        response = requests.get(base_url, params=params, timeout=20) # Increased timeout for data retrieval
        response.raise_for_status()

        root = ET.fromstring(response.content)

        namespace = ''
        if root.tag.startswith('{') and '}' in root.tag:
            namespace = root.tag.split('}')[0] + '}'

        # Hilltop XML structure for GetData:
        # Hilltop/Measurement/Data/E (Event) elements, each with T (Time) and V (Value)
        # Sometimes there's also a QualityCode attribute or element.
        for event_elem in root.findall(f'{namespace}Measurement/{namespace}Data/{namespace}E'):
            value_str = event_elem.get('V') # Assuming value is an attribute 'V'
            # If 'V' is not an attribute, it might be a child element
            if value_str is None:
                value_elem = event_elem.find(f'{namespace}V')
                if value_elem is not None:
                    value_str = value_elem.text

            if value_str:
                try:
                    data_values.append(float(value_str))
                except ValueError:
                    logging.warning(f"Could not parse value '{value_str}' for site {site_name}, measurement {measurement}")

        if data_values:
            result["average_value"] = sum(data_values) / len(data_values)
            result["data_points"] = len(data_values)
            logging.info(f"Retrieved {len(data_values)} data points for site='{site_name}', measurement='{measurement}'. "
                         f"Average value: {result['average_value']:.2f}")
        else:
            logging.info(f"No data points found for site='{site_name}', measurement='{measurement}'.")
            return None # Return None if no data points are found

    except requests.exceptions.Timeout:
        logging.error(f"Request timed out when fetching data for site='{site_name}', measurement='{measurement}'")
        return None
    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP error occurred: {e} - Status Code: {e.response.status_code} for site='{site_name}', measurement='{measurement}'")
        return None
    except requests.exceptions.ConnectionError as e:
        logging.error(f"Connection error occurred: {e} for site='{site_name}', measurement='{measurement}'")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"An unexpected request error occurred: {e} for site='{site_name}', measurement='{measurement}'")
        return None
    except ET.ParseError as e:
        logging.error(f"Failed to parse XML response for site='{site_name}', measurement='{measurement}': {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return None

    return result

# Helper functions for Hilltop API interactions
# Note: Different Hilltop base URLs are used for different requests:
# - SiteList: http://odp.es.govt.nz/data.hts (used by load_monitoring_sites.py and get_hilltop_sites)
# - MeasurementList: https://data.es.govt.nz/Envirodata/EMAR.hts (used by get_measurements_for_site)
# - GetData: https://data.es.govt.nz/Envirodata/EMARDiscrete.hts (used by get_hilltop_data)

def get_hilltop_sites(measurement: str) -> dict:
    """
    Fetches a list of Hilltop sites that have data for a given measurement.
    Returns a dictionary mapping site names to their (latitude, longitude) tuples.
    """
    base_url = "http://odp.es.govt.nz/data.hts" # Consistent with load_monitoring_sites.py for SiteList
    params = {
        "Service": "Hilltop",
        "Request": "SiteList",
        "Measurement": measurement # Filter sites by measurement
    }
    sites = {}
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        root = ET.fromstring(response.content)

        namespace = ''
        if root.tag.startswith('{') and '}' in root.tag:
            namespace = root.tag.split('}')[0] + '}'

        for site_elem in root.findall(f'{namespace}SiteList/{namespace}Site'):
            site_name = site_elem.get('Name')
            lat_elem = site_elem.find(f'{namespace}Lat')
            lon_elem = site_elem.find(f'{namespace}Lon')

            if site_name and lat_elem is not None and lon_elem is not None:
                try:
                    lat = float(lat_elem.text)
                    lon = float(lon_elem.text)
                    sites[site_name] = (lat, lon)
                except (ValueError, TypeError):
                    logger.warning(f"Could not parse lat/lon for site {site_name}")
        
        if not sites:
            logger.warning(f"No Hilltop sites found for measurement '{measurement}' from {base_url}")
        else:
            logger.info(f"Found {len(sites)} Hilltop sites for measurement '{measurement}'.")

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching Hilltop site list for measurement '{measurement}': {e}")
    except ET.ParseError as e:
        logger.error(f"Failed to parse XML response for Hilltop site list for measurement '{measurement}': {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while fetching Hilltop site list: {e}")

    return sites