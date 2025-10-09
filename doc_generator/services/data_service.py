import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, date
import logging
import random
import math
import urllib.parse
import time
from django.conf import settings
from django.core.cache import cache
from bs4 import BeautifulSoup
import pandas as pd
import json
from typing import List
from doc_generator.models import HistoricalMeasurement

logger = logging.getLogger(__name__)

# This variable will hold the singleton instance of the kHilltopConnector.
_hilltop_connector = None

# NOTE: All Hilltop-related functions are temporarily disabled as the endpoints are offline.
# They are kept here for future reference if the service is restored.

# --- DATA SERVICE CLASS ---

class DataService:
    """
    A service for fetching water quality data from Environment Southland's Hilltop server
    with a resilient fallback chain.
    """
    HILLTOP_MEASUREMENT_MAPPING = {
        # This mapping is crucial for the scraping fallback. The keys are what the view requests,
        # and the values are what the scraping function will find on the envdata.es.govt.nz website.
        "E-Coli <CFU>": "E-Coli <CFU>",
        "Nitrogen (Nitrate Nitrite)": "Nitrogen (Nitrate Nitrite)",
        "Phosphorus (Dissolved Reactive)": "Phosphorus (Dissolved Reactive)",
        "Turbidity (FNU)": "Turbidity (FNU)",
    }

    LAWA_MEASUREMENT_MAPPING = {
        "E-Coli <CFU>": "ECOLI",
        "Nitrogen (Nitrate Nitrite)": "NNN",
        "Phosphorus (Dissolved Reactive)": "DRP",
        "Turbidity (FNU)": "TURB",
    }

    def get_groundwater_data(self, site) -> dict:
        """
        Fetches data for a single groundwater well from the LAWA API.
        """
        if not site or site.site_type != 'groundwater':
            return None

        cache_key = f"groundwater_data:{site.hilltop_site_id}"
        cached_data = cache.get(cache_key)
        if cached_data:
            logger.info(f"Cache HIT for groundwater site {site.hilltop_site_id}. Returning cached data.")
            return cached_data

        logger.info(f"--- Fetching groundwater data for site: {site.site_name} (ID: {site.hilltop_site_id}) ---")
        
        url = f"https://www.lawa.org.nz/api/well/{site.hilltop_site_id}?format=json"
        
        try:
            response = self.session.get(url, timeout=20)
            response.raise_for_status()
            api_data = response.json()
        except (requests.RequestException, json.JSONDecodeError) as e:
            logger.error(f"Failed to fetch or parse groundwater data for site {site.hilltop_site_id}: {e}")
            return None

        # Process the returned data to structure it consistently
        processed_data = {}
        measurements = api_data.get('measurements', [])
        for measurement in measurements:
            measurement_name = measurement.get('name')
            if not measurement_name:
                continue

            # Find the latest value from the 'chartData'
            chart_data = measurement.get('chartData', [])
            latest_value = None
            if chart_data and isinstance(chart_data, list) and len(chart_data) > 0:
                # Assuming the last item is the most recent
                latest_point = chart_data[-1]
                if isinstance(latest_point, list) and len(latest_point) > 1:
                    latest_value = latest_point[1]

            processed_data[measurement_name] = {
                'latest_value': latest_value,
                'unit': measurement.get('unit'),
            }

        cache.set(cache_key, processed_data, timeout=7200) # Cache for 2 hours
        return processed_data

    def __init__(self):
        """Initializes the DataService with a requests session with retry logic."""
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504]
        )
        # Add a standard User-Agent header to avoid being blocked by services like LAWA.
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.session.headers.update(headers)
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def get_water_quality_data(self, nearby_sites: list, measurement: str, lat: float, lon: float) -> dict:
        """
        Orchestrates fetching water quality data by trying a chain of methods
        for each nearby site until data is found.
        """
        for site in nearby_sites:
            logger.info(f"--- Trying site: {site.site_name} for measurement: {measurement} ---")
            cache_key = f"water_quality:{site.hilltop_site_id}:{measurement}:{random.randint(1, 10000)}"
            cached_data = cache.get(cache_key)
            if cached_data:
                logger.info(f"Cache HIT for site {site.site_name}. Returning cached data.")
                return cached_data

            # Scrape-first fallback chain. The scraper now gets all measurements.
            # The measurement parameter is ignored by the scraper but used by fallbacks.
            data = self._get_data_from_lawa_scrape(site, measurement)
            # Hilltop is disabled, so we log and skip to the next step.
            if not data:
                logger.info(f"LAWA scrape failed for site {site.site_name}. Hilltop is temporarily disabled. No live data available.")
                # data = self._get_data_with_khilltop(site, measurement) # This is disabled.

            if data: # If LAWA scrape succeeded
                cache.set(cache_key, data, timeout=7200) # Cache for 2 hours
                return data

        # If all sites and fallbacks fail
        logger.warning(f"Exhausted all live data sources for measurement '{measurement}'. Trying historical database fallback.")
        
        # Final fallback to the historical database
        historical_data = self._get_data_from_historical_db(nearby_sites[0], measurement)
        if historical_data:
            return historical_data

        logger.error(f"All data sources failed for measurement '{measurement}', including historical DB. No data to return.")
        return {"error": "No data available from any source", "source": "None"}

    def _get_data_with_khilltop(self, site, measurement):
        """
        Secondary fallback. Fetches data directly from the ES Hilltop server.
        NOTE: This method is temporarily disabled as the Hilltop endpoints are offline.
        """
        logger.warning(f"Skipping Hilltop request for site '{site.hilltop_site_id}' as it is temporarily disabled.")
        return None

    def _get_data_from_lawa_surface_water(self, site, measurement):
        """
        Fetches data for a single surface water site from the LAWA API.
        NOTE: This method is deprecated as the Umbraco API endpoint is no longer active.
        It is kept for historical reference but will fail gracefully.
        """
        logger.warning(f"Attempting to use deprecated LAWA API for site '{site.hilltop_site_id}'. This will fail.")
        return None

    @staticmethod
    def _retry_scrape(func):
        """A decorator to add retry logic with exponential backoff for scraping."""
        def wrapper(*args, **kwargs):
            self_instance = args[0]
            site = args[1]
            for attempt in range(3): # Retry up to 3 times
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.HTTPError as e:
                    if 400 <= e.response.status_code < 500:
                        logger.warning(f"LAWA Scrape: Client error {e.response.status_code} for site {site.site_name}. Not retrying.")
                        return None # Don't retry on 4xx errors (like 404 Not Found)
                    logger.error(f"LAWA Scrape: HTTP error on attempt {attempt + 1} for site {site.site_name}: {e}. Retrying...")
                except requests.exceptions.RequestException as e:
                    logger.error(f"LAWA Scrape: Request error on attempt {attempt + 1} for site {site.site_name}: {e}. Retrying...")
                
                time.sleep(2 ** attempt) # Exponential backoff: 1s, 2s, 4s
            
            logger.error(f"LAWA Scrape: All 3 attempts failed for site {site.site_name}.")
            return None
        return wrapper

    @_retry_scrape
    def _get_data_from_lawa_scrape(self, site, measurement):
        """
        Primary data fetching method. Fetches data from the new LAWA API.
        """
        if not site.lawa_id:
            logger.warning(f"LAWA Scrape: Site '{site.site_name}' has no lawa_id.")
            return None

        indicator = self.LAWA_MEASUREMENT_MAPPING.get(measurement)
        if not indicator:
            logger.warning(f"LAWA Scrape: No indicator mapping found for measurement '{measurement}'.")
            return None

        logger.info(f"Attempting LAWA API request for site: '{site.lawa_id}' with indicator: '{indicator}'")

        end_year = datetime.now().year
        start_year = end_year - 10

        url = f"https://www.lawa.org.nz/umbraco/api/riverservice/GetRecGraphData?location={site.lawa_id}&indicator={indicator}&startYear={start_year}&endYear={end_year}"

        logger.info(f"Fetching data from URL: {url}")

        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        raw_data = response.json()

        readings = []
        if isinstance(raw_data, dict) and 'Data' in raw_data:
            data_list = raw_data['Data']
            for item in data_list:
                if item.get('data') and item['data']:
                    ts, val = item['data'][0]
                    dt = datetime.fromtimestamp(ts / 1000)
                    readings.append((dt, val))
        elif isinstance(raw_data, list) and raw_data and isinstance(raw_data[0], dict) and 'data' in raw_data[0]:
            # Handle [{"data": [[timestamp, value]]}]
            for item in raw_data:
                if item.get('data') and item['data']:
                    ts, val = item['data'][0]
                    dt = datetime.fromtimestamp(ts / 1000)
                    readings.append((dt, val))
        elif isinstance(raw_data, list) and raw_data and isinstance(raw_data[0], list):
            # Handle [[timestamp, value]]
            for ts, val in raw_data:
                dt = datetime.fromtimestamp(ts / 1000)
                readings.append((dt, val))

        if not readings:
            logger.warning(f"LAWA API: No processable data returned for site '{site.lawa_id}' and indicator '{indicator}'.")
            return None

        # Sort by date to ensure the last one is the latest
        readings.sort(key=lambda x: x[0])

        latest_date, latest_value = readings[-1]

        historical_data = [(dt.strftime('%Y-%m-%d'), val) for dt, val in readings]

        return {
            'source': 'LAWA API',
            'data': {
                measurement: {
                    'latest_result': float(latest_value) if latest_value is not None else None,
                    'latest_date': latest_date.strftime('%Y-%m-%d') if latest_date else None,
                    'historical_data': historical_data,
                    'five_year_median': None, # Not available in new API
                    'state': None, # Not available in new API
                    'trend': None, # Not available in new API
                }
            },
            'site_name': site.site_name
        }

    def _get_data_from_historical_db(self, site, measurement):
        """
        Final fallback method. Queries the local HistoricalMeasurement table for the
        most recent record for a given site and measurement.
        """
        logger.info(f"Attempting fallback to local historical DB for site '{site.hilltop_site_id}'")
        try:
            # Find the most recent historical record for this site and measurement
            latest_measurement = HistoricalMeasurement.objects.filter(
                council_site_id=site.hilltop_site_id,
                measurement=measurement
            ).order_by('-timestamp').first()

            if latest_measurement:
                logger.info(f"Historical DB: Found data for '{measurement}' from {latest_measurement.timestamp}.")
                return {
                    'average_value': latest_measurement.value,
                    'data_points': 1,
                    'source': f"Historical DB ({latest_measurement.timestamp.year})",
                    'is_historical': True,
                    'site_name': site.site_name,
                }
            return None
        except Exception as e:
            logger.error(f"Historical DB lookup failed for site '{site.hilltop_site_id}': {e}")
            return None