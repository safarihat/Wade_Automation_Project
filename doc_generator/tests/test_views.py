from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from django.contrib.gis.geos import Point
from doc_generator.models import MonitoringSite
from unittest.mock import patch, Mock
import json

class WaterQualityViewTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='testuser', password='testpassword')
        self.client.login(username='testuser', password='testpassword')

        # Create some MonitoringSite instances for GeoDjango queries
        self.site1 = MonitoringSite.objects.create(
            site_name="TestSite1",
            hilltop_site_id="TS1",
            location=Point(170.0, -45.0, srid=4326) # Near user_lat, lon
        )
        self.site2 = MonitoringSite.objects.create(
            site_name="TestSite2",
            hilltop_site_id="TS2",
            location=Point(170.5, -45.5, srid=4326) # Further away
        )
        self.site3 = MonitoringSite.objects.create(
            site_name="TestSite3",
            hilltop_site_id="TS3",
            location=Point(171.0, -46.0, srid=4326) # Even further
        )

        self.url = reverse('doc_generator:api_get_water_quality_data')
        self.user_lat = -45.001
        self.user_lon = 170.001
        self.measurements = [
            "E-Coli <CFU>",
            "Nitrogen (Nitrate Nitrite)",
            "Phosphorus (Dissolved Reactive)",
            "Turbidity (FNU)",
        ]

    def test_api_get_water_quality_data_not_logged_in(self):
        self.client.logout()
        response = self.client.get(self.url, {'lat': self.user_lat, 'lon': self.user_lon})
        self.assertEqual(response.status_code, 302) # Redirect to login

    def test_api_get_water_quality_data_missing_params(self):
        response = self.client.get(self.url, {'lat': self.user_lat})
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Missing 'lat' or 'lon' parameters.", response.content)

        response = self.client.get(self.url, {'lon': self.user_lon})
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Missing 'lat' or 'lon' parameters.", response.content)

    def test_api_get_water_quality_data_invalid_params(self):
        response = self.client.get(self.url, {'lat': 'abc', 'lon': self.user_lon})
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Invalid 'lat' or 'lon' parameters.", response.content)

        response = self.client.get(self.url, {'lat': self.user_lat, 'lon': 'xyz'})
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Invalid 'lat' or 'lon' parameters.", response.content)

    @patch('doc_generator.services.data_service.DataService.get_water_quality_data')
    def test_api_get_water_quality_data_success(self, mock_get_water_quality_data):
        # Mock DataService to return successful Hilltop data
        mock_get_water_quality_data.return_value = {
            "average_value": 10.0,
            "data_points": 5,
            "source": "Hilltop",
        }

        response = self.client.get(self.url, {'lat': self.user_lat, 'lon': self.user_lon})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)

        self.assertEqual(data['site_name'], self.site1.site_name)
        self.assertAlmostEqual(data['distance_km'], self.site1.location.distance(Point(self.user_lon, self.user_lat, srid=4326)).km, places=2)
        self.assertIn("E-Coli <CFU>", data['data'])
        self.assertEqual(data['data']["E-Coli <CFU>"]['average_value'], 10.0)
        self.assertEqual(data['data']["E-Coli <CFU>"]['source'], "Hilltop")
        
        # Ensure DataService was called for each measurement with the correct hilltop_site_id
        self.assertEqual(mock_get_water_quality_data.call_count, len(self.measurements))
        for measurement in self.measurements:
            mock_get_water_quality_data.assert_any_call(self.site1.hilltop_site_id, measurement, self.user_lat, self.user_lon)

    @patch('doc_generator.services.data_service.DataService.get_water_quality_data')
    def test_api_get_water_quality_data_no_monitoring_sites(self, mock_get_water_quality_data):
        # Delete all monitoring sites to simulate no sites found
        MonitoringSite.objects.all().delete()

        response = self.client.get(self.url, {'lat': self.user_lat, 'lon': self.user_lon})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)

        self.assertEqual(data['site_name'], "No Site Found")
        self.assertIsNone(data['distance_km'])
        self.assertIn("E-Coli <CFU>", data['data'])
        self.assertEqual(data['data']["E-Coli <CFU>"]['source'], "mock")
        self.assertEqual(mock_get_water_quality_data.call_count, len(self.measurements)) # Still calls for mock data

    @patch('doc_generator.services.data_service.DataService.get_water_quality_data')
    def test_api_get_water_quality_data_partial_failure(self, mock_get_water_quality_data):
        # Mock DataService to return success for one measurement, failure for another
        def side_effect_func(hilltop_site_id, measurement, lat, lon):
            if measurement == "E-Coli <CFU>":
                return {"average_value": 15.0, "data_points": 3, "source": "Hilltop"}
            else:
                return None # Simulate failure for other measurements

        mock_get_water_quality_data.side_effect = side_effect_func

        response = self.client.get(self.url, {'lat': self.user_lat, 'lon': self.user_lon})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)

        self.assertEqual(data['site_name'], self.site1.site_name)
        self.assertEqual(data['data']["E-Coli <CFU>"]['average_value'], 15.0)
        self.assertEqual(data['data']["E-Coli <CFU>"]['source'], "Hilltop")
        self.assertEqual(data['data']["Nitrogen (Nitrate Nitrite)"]['source'], "Failed to retrieve")
        self.assertEqual(mock_get_water_quality_data.call_count, len(self.measurements))

    @patch('doc_generator.services.data_service.DataService.get_water_quality_data')
    def test_api_get_water_quality_data_fallback_render(self, mock_get_water_quality_data):
        # Mock DataService to return fallback data
        mock_get_water_quality_data.return_value = {
            "average_value": 75.0,
            "data_points": 8,
            "source": "Fallback: ArcGIS",
        }

        response = self.client.get(self.url, {'lat': self.user_lat, 'lon': self.user_lon})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)

        self.assertEqual(data['site_name'], self.site1.site_name)
        self.assertIn("E-Coli <CFU>", data['data'])
        self.assertEqual(data['data']["E-Coli <CFU>"]['average_value'], 75.0)
        self.assertEqual(data['data']["E-Coli <CFU>"]['source'], "Fallback: ArcGIS")
        self.assertEqual(mock_get_water_quality_data.call_count, len(self.measurements))
