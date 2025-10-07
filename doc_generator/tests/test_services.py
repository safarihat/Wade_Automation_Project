import unittest
from unittest.mock import patch, Mock
from django.test import TestCase
from doc_generator.services.data_service import DataService, get_hilltop_sites, get_measurements_for_site, get_hilltop_data
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import json

# Mock XML responses for Hilltop API
MOCK_HILLTOP_SITELIST_XML = """
<Hilltop>
    <SiteList>
        <Site Name="SiteA">
            <Lat> -45.0 </Lat>
            <Lon> 170.0 </Lon>
        </Site>
        <Site Name="SiteB">
            <Lat> -45.1 </Lat>
            <Lon> 170.1 </Lon>
        </Site>
        <Site Name="SiteC">
            <Lat> -45.2 </Lat>
            <Lon> 170.2 </Lon>
        </Site>
    </SiteList>
</Hilltop>
"""

MOCK_HILLTOP_MEASUREMENTLIST_XML = """
<Hilltop>
    <MeasurementList>
        <Measurement>
            <DataSource>
                <MeasurementName>E-Coli &lt;CFU&gt;</MeasurementName>
            </DataSource>
        </Measurement>
        <Measurement>
            <DataSource>
                <MeasurementName>Nitrogen (Nitrate Nitrite)</MeasurementName>
            </DataSource>
        </Measurement>
    </MeasurementList>
</Hilltop>
"""

MOCK_HILLTOP_GETDATA_XML = """
<Hilltop>
    <Measurement>
        <Data>
            <E V="10.5" T="2023-01-01T00:00:00"/>
            <E V="12.3" T="2023-01-02T00:00:00"/>
            <E V="11.0" T="2023-01-03T00:00:00"/>
        </Data>
    </Measurement>
</Hilltop>
"""

MOCK_HILLTOP_GETDATA_EMPTY_XML = """
<Hilltop>
    <Measurement>
        <Data>
        </Data>
    </Measurement>
</Hilltop>
"""

# Mock LAWA API response
MOCK_LAWA_RESPONSE = [
    {"Measurement": "E-Coli <CFU>", "Value": 10},
    {"Measurement": "E-Coli <CFU>", "Value": 20},
    {"Measurement": "Nitrogen (Nitrate Nitrite)", "Value": 1.5},
]

MOCK_LAWA_EMPTY_RESPONSE = []

# Mock ArcGIS API response
MOCK_ARCGIS_RESPONSE = {
    "features": [{"attributes": {"value": 100}}],
}

MOCK_ARCGIS_EMPTY_RESPONSE = {
    "features": [],
}


class MockResponse:
    def __init__(self, content="", status_code=200, json_data=None, headers=None):
        self._content = content
        self.status_code = status_code
        self._json_data = json_data
        self.headers = headers or {}
        self.request = Mock() # Mock the request object

    @property
    def content(self):
        return self._content.encode('utf-8')

    def json(self):
        if self._json_data is not None:
            return self._json_data
        try:
            return json.loads(self._content)
        except json.JSONDecodeError:
            raise ValueError("No JSON data provided for mock response and content is not valid JSON")

    def raise_for_status(self):
        if self.status_code >= 400:
            response_mock = Mock()
            response_mock.status_code = self.status_code
            response_mock.text = self._content if self._content else ""
            raise requests.exceptions.HTTPError(f"HTTP Error: {self.status_code}", response=response_mock)


class DataServiceTests(TestCase):

    def setUp(self):
        self.data_service = DataService()
        self.hilltop_site_id = "Aparima River at Thornbury" # Use a site with a LAWA mapping
        self.measurement = "E-Coli <CFU>"
        self.lat = -45.0
        self.lon = 170.0

    @patch('requests.get')
    def test_get_hilltop_sites_success(self, mock_get):
        mock_get.return_value = MockResponse(MOCK_HILLTOP_SITELIST_XML)
        sites = get_hilltop_sites(self.measurement)
        self.assertIn("SiteA", sites)
        self.assertEqual(sites["SiteA"], (-45.0, 170.0))
        self.assertEqual(len(sites), 3)
        mock_get.assert_called_once()

    @patch('requests.get')
    def test_get_hilltop_sites_no_sites(self, mock_get):
        mock_get.return_value = MockResponse("<Hilltop><SiteList></SiteList></Hilltop>")
        sites = get_hilltop_sites(self.measurement)
        self.assertEqual(sites, {})
        mock_get.assert_called_once()

    @patch('requests.get')
    def test_get_hilltop_sites_api_error(self, mock_get):
        mock_get.side_effect = requests.exceptions.RequestException("Test Error")
        sites = get_hilltop_sites(self.measurement)
        self.assertEqual(sites, {})
        mock_get.assert_called_once()

    @patch('requests.get')
    def test_get_measurements_for_site_success(self, mock_get):
        mock_get.return_value = MockResponse(MOCK_HILLTOP_MEASUREMENTLIST_XML)
        measurements = get_measurements_for_site(self.hilltop_site_id)
        self.assertIn("E-Coli <CFU>", measurements)
        self.assertEqual(len(measurements), 2)
        mock_get.assert_called_once()

    @patch('requests.get')
    def test_get_measurements_for_site_no_measurements(self, mock_get):
        mock_get.return_value = MockResponse("<Hilltop><MeasurementList></MeasurementList></Hilltop>")
        measurements = get_measurements_for_site(self.hilltop_site_id)
        self.assertEqual(measurements, [])
        mock_get.assert_called_once()

    @patch('requests.get')
    def test_get_measurements_for_site_api_error(self, mock_get):
        mock_get.side_effect = requests.exceptions.RequestException("Test Error")
        measurements = get_measurements_for_site(self.hilltop_site_id)
        self.assertEqual(measurements, [])
        mock_get.assert_called_once()

    @patch('requests.get')
    def test_get_hilltop_data_success(self, mock_get):
        mock_get.return_value = MockResponse(MOCK_HILLTOP_GETDATA_XML)
        data = get_hilltop_data(self.hilltop_site_id, self.measurement)
        self.assertIsNotNone(data)
        self.assertAlmostEqual(data['average_value'], (10.5 + 12.3 + 11.0) / 3)
        self.assertEqual(data['data_points'], 3)
        self.assertEqual(data['source'], 'Hilltop')
        mock_get.assert_called_once()

    @patch('requests.get')
    def test_get_hilltop_data_no_data(self, mock_get):
        mock_get.return_value = MockResponse(MOCK_HILLTOP_GETDATA_EMPTY_XML)
        data = get_hilltop_data(self.hilltop_site_id, self.measurement)
        self.assertIsNone(data)
        mock_get.assert_called_once()

    @patch('requests.get')
    def test_get_hilltop_data_api_error(self, mock_get):
        mock_get.side_effect = requests.exceptions.RequestException("Test Error")
        data = get_hilltop_data(self.hilltop_site_id, self.measurement)
        self.assertIsNone(data)
        mock_get.assert_called_once()

    @patch('doc_generator.services.data_service.get_hilltop_data')
    @patch('doc_generator.services.data_service.DataService._get_from_es_arcgis')
    @patch('doc_generator.services.data_service.DataService._get_from_lawa')
    @patch('doc_generator.services.data_service.DataService._get_mock_data')
    def test_get_water_quality_data_hilltop_success(self, mock_mock, mock_lawa, mock_arcgis, mock_hilltop_data):
        # Mock Hilltop data to be successful
        mock_hilltop_data.return_value = {"average_value": 15.0, "data_points": 5, "source": "Hilltop"}
        
        # Ensure LAWA and ArcGIS are not called
        mock_lawa.return_value = {"error": "LAWA failed"}
        mock_arcgis.return_value = {"error": "ArcGIS failed"}

        result = self.data_service.get_water_quality_data(self.hilltop_site_id, self.measurement, self.lat, self.lon)
        
        self.assertEqual(result['source'], 'Hilltop')
        mock_lawa.assert_called_once() # LAWA is tried first
        mock_arcgis.assert_called_once() # ArcGIS is tried second
        mock_hilltop_data.assert_called_once_with(self.hilltop_site_id, self.measurement, unittest.mock.ANY, unittest.mock.ANY)
        mock_mock.assert_not_called()

    @patch('doc_generator.services.data_service.get_hilltop_data')
    @patch('doc_generator.services.data_service.DataService._get_from_es_arcgis')
    @patch('doc_generator.services.data_service.DataService._get_from_lawa')
    @patch('doc_generator.services.data_service.DataService._get_mock_data')
    def test_get_water_quality_data_arcgis_fallback(self, mock_mock, mock_lawa, mock_arcgis, mock_hilltop_data):
        # Mock Hilltop to fail
        mock_hilltop_data.return_value = None
        
        # Mock LAWA to fail with an HTTP error
        mock_lawa.return_value = {"error": "LAWA failed with HTTP error", "http_error": True, "status_code": 404}

        # Mock ArcGIS to be successful
        mock_arcgis.return_value = {"average_value": 100.0, "data_points": 10, "source": "ArcGIS"}

        result = self.data_service.get_water_quality_data(self.hilltop_site_id, self.measurement, self.lat, self.lon)
        
        self.assertEqual(result['source'], 'ArcGIS')
        mock_lawa.assert_called_once()
        mock_arcgis.assert_called_once()
        mock_hilltop_data.assert_called_once()
        mock_mock.assert_not_called()

    @patch('doc_generator.services.data_service.get_hilltop_data')
    @patch('doc_generator.services.data_service.DataService._get_from_es_arcgis')
    @patch('doc_generator.services.data_service.DataService._get_from_lawa')
    @patch('doc_generator.services.data_service.DataService._get_mock_data')
    def test_get_water_quality_data_mock_fallback(self, mock_mock, mock_lawa, mock_arcgis, mock_hilltop_data):
        # Mock all to fail
        mock_hilltop_data.return_value = None
        mock_lawa.return_value = {"error": "LAWA failed with HTTP error", "http_error": True, "status_code": 404}
        mock_arcgis.return_value = {"error": "ArcGIS failed"}
        
        # Mock mock data to be successful
        mock_mock.return_value = {"average_value": 50.0, "data_points": 1, "source": "mock"}

        result = self.data_service.get_water_quality_data(self.hilltop_site_id, self.measurement, self.lat, self.lon)
        
        self.assertEqual(result['source'], 'mock')
        mock_lawa.assert_called_once()
        mock_arcgis.assert_called_once()
        mock_hilltop_data.assert_called_once()
        mock_mock.assert_called_once()

    @patch('doc_generator.services.data_service.get_hilltop_data')
    @patch('doc_generator.services.data_service.DataService._get_from_es_arcgis')
    @patch('doc_generator.services.data_service.DataService._get_from_lawa')
    @patch('doc_generator.services.data_service.DataService._get_mock_data')
    def test_get_from_lawa_404_fallback(self, mock_mock, mock_lawa, mock_arcgis, mock_hilltop_data):
        # Simulate LAWA returning a 404 HTTP error
        mock_lawa.return_value = {"error": "LAWA 404", "source": "LAWA", "http_error": True, "status_code": 404}
        # Simulate ArcGIS returning valid data
        mock_arcgis.return_value = {"average_value": 75.0, "data_points": 8, "source": "Fallback: ArcGIS"}
        # Ensure Hilltop and mock are not called initially
        mock_hilltop_data.return_value = None
        mock_mock.return_value = None

        result = self.data_service.get_water_quality_data(self.hilltop_site_id, self.measurement, self.lat, self.lon)

        self.assertEqual(result['source'], 'Fallback: ArcGIS')
        self.assertEqual(result['average_value'], 75.0)
        mock_lawa.assert_called_once()
        mock_arcgis.assert_called_once()
        mock_hilltop_data.assert_not_called()
        mock_mock.assert_not_called()

    @patch('requests.get')
    def test_get_from_lawa_success(self, mock_get):
        mock_get.return_value = MockResponse(content="", json_data=MOCK_LAWA_RESPONSE)
        result = self.data_service._get_from_lawa(self.hilltop_site_id, self.measurement)
        self.assertAlmostEqual(result['average_value'], 15.0)
        self.assertEqual(result['data_points'], 2)
        self.assertEqual(result['source'], 'LAWA')
        mock_get.assert_called_once_with(
            'https://www.lawa.org.nz/GetMeasurementsBySiteID?siteID=12345',
            timeout=10
        )

    @patch('requests.get')
    def test_get_from_lawa_http_error(self, mock_get):
        # Simulate a 404 Not Found error from LAWA
        mock_get.return_value = MockResponse(content="Not Found", status_code=404)
        result = self.data_service._get_from_lawa(self.hilltop_site_id, self.measurement)
        self.assertIn('error', result)
        self.assertEqual(result['source'], 'LAWA')
        self.assertTrue(result['http_error'])
        self.assertEqual(result['status_code'], 404)

    @patch('requests.get')
    def test_get_from_lawa_no_data(self, mock_get):
        mock_get.return_value = MockResponse(content="", json_data=MOCK_LAWA_EMPTY_RESPONSE)
        result = self.data_service._get_from_lawa(self.hilltop_site_id, self.measurement)
        self.assertIn('error', result)
        self.assertEqual(result['source'], 'LAWA')

    @patch('requests.get')
    def test_get_from_lawa_api_error(self, mock_get):
        mock_get.side_effect = requests.exceptions.RequestException("LAWA Test Error")
        result = self.data_service._get_from_lawa(self.hilltop_site_id, self.measurement)
        self.assertIn('error', result)
        self.assertEqual(result['source'], 'LAWA')

    @patch('requests.get')
    def test_get_from_es_arcgis_success(self, mock_get):
        mock_get.return_value = MockResponse(content="", json_data=MOCK_ARCGIS_RESPONSE)
        result = self.data_service._get_from_es_arcgis(self.lat, self.lon, self.measurement)
        self.assertEqual(result['average_value'], 110.0) # Placeholder value in _get_from_es_arcgis
        self.assertEqual(result['data_points'], 1)
        self.assertEqual(result['source'], 'ArcGIS')

    @patch('requests.get')
    def test_get_from_es_arcgis_no_data(self, mock_get):
        mock_get.return_value = MockResponse(content="", json_data=MOCK_ARCGIS_EMPTY_RESPONSE)
        result = self.data_service._get_from_es_arcgis(self.lat, self.lon, self.measurement)
        self.assertIn('error', result)
        self.assertEqual(result['source'], 'ArcGIS')

    @patch('requests.get')
    def test_get_from_es_arcgis_api_error(self, mock_get):
        mock_get.side_effect = requests.exceptions.RequestException("ArcGIS Test Error")
        result = self.data_service._get_from_es_arcgis(self.lat, self.lon, self.measurement)
        self.assertIn('error', result)
        self.assertEqual(result['source'], 'ArcGIS')

    @patch('doc_generator.services.data_service.get_hilltop_data')
    @patch('doc_generator.services.data_service.DataService._get_from_es_arcgis')
    def test_get_from_hilltop_success(self, mock_arcgis, mock_hilltop_data):
        mock_hilltop_data.return_value = {"average_value": 20.0, "data_points": 7, "source": "Hilltop"}
        result = self.data_service._get_from_hilltop(self.hilltop_site_id, self.measurement, self.lat, self.lon)
        self.assertEqual(result['source'], 'Hilltop')
        self.assertEqual(result['nearest_site_name'], self.hilltop_site_id)
        self.assertEqual(result['distance_km'], 0.0)
        mock_hilltop_data.assert_called_once()
        mock_arcgis.assert_not_called()

    @patch('doc_generator.services.data_service.get_hilltop_data')
    @patch('doc_generator.services.data_service.DataService._get_from_es_arcgis')
    def test_get_from_hilltop_no_data_fallback(self, mock_arcgis, mock_hilltop_data):
        mock_hilltop_data.return_value = None # Simulate no data from Hilltop
        mock_arcgis.return_value = {"average_value": 100.0, "data_points": 10, "source": "ArcGIS"}
        result = self.data_service._get_from_hilltop(self.hilltop_site_id, self.measurement, self.lat, self.lon)
        self.assertEqual(result['source'], 'ArcGIS')
        mock_hilltop_data.assert_called_once()
        mock_arcgis.assert_called_once()

    @patch('doc_generator.services.data_service.get_hilltop_data')
    @patch('doc_generator.services.data_service.DataService._get_from_es_arcgis')
    def test_get_from_hilltop_api_error_fallback(self, mock_arcgis, mock_hilltop_data):
        mock_hilltop_data.side_effect = requests.exceptions.RequestException("Hilltop Test Error")
        mock_arcgis.return_value = {"average_value": 100.0, "data_points": 10, "source": "ArcGIS"}
        result = self.data_service._get_from_hilltop(self.hilltop_site_id, self.measurement, self.lat, self.lon)
        self.assertEqual(result['source'], 'ArcGIS')
        mock_hilltop_data.assert_called_once()
        mock_arcgis.assert_called_once()
