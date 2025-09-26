from django.test import TestCase
from unittest.mock import patch, MagicMock
from doc_generator.services.arcgis_service import ArcGISService

class ArcGISServiceTests(TestCase):
    """
    Tests for the ArcGISService to ensure data aggregation logic is correct.
    """

    @patch('doc_generator.services.arcgis_service.requests.get')
    def test_aggregate_results_logic(self, mock_get):
        """
        Tests the _aggregate_results method to ensure it correctly identifies
        the primary catchment, calculates area, and summarizes degradation.
        """
        # Mock the API response to return a predictable set of features
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'features': [
                {'attributes': {'nzsegment': 12345, 'Degradation_TP': 'High', 'shape_area': 100000}},
                {'attributes': {'nzsegment': 12345, 'Degradation_Ecoli': 'Medium', 'shape_area': 100000}},
                {'attributes': {'nzsegment': 67890, 'Degradation_TP': 'Low', 'shape_area': 50000}},
            ]
        }
        mock_get.return_value = mock_response

        # Initialize the service with dummy coordinates
        service = ArcGISService(lat=-46.28, lon=168.02)
        
        # The raw attributes that would be collected by get_catchment_data
        # We simulate the collected data to isolate the aggregation logic for testing
        all_attributes = [
            {'nzsegment': 12345, 'shape_area': 100000, 'layer': 'TP', 'Degradation_TP': 'High'},
            {'nzsegment': 12345, 'shape_area': 100000, 'layer': 'Ecoli', 'Degradation_Ecoli': 'Medium'},
            {'nzsegment': 67890, 'shape_area': 50000, 'layer': 'TP', 'Degradation_TP': 'Low'},
            {'nzsegment': 12345, 'shape_area': 100000, 'layer': 'TN', 'Degradation_TN': 'Very High'},
        ]

        # Call the aggregation method directly
        result = service._aggregate_results(all_attributes)

        # Assertions
        self.assertIsNotNone(result)
        self.assertEqual(result['primary_nzsegment'], 12345)
        self.assertEqual(result['area_ha'], 10.0)  # 100,000 m^2 = 10 ha
        self.assertEqual(len(result['degradation_summary']), 3)
        self.assertEqual(result['degradation_summary']['TP'], 'High')
        self.assertEqual(result['degradation_summary']['Ecoli'], 'Medium')
        self.assertEqual(result['degradation_summary']['TN'], 'Very High')

    def test_aggregate_results_no_attributes(self):
        """
        Tests that _aggregate_results returns None if no attributes are provided.
        """
        service = ArcGISService(lat=-46.28, lon=168.02)
        result = service._aggregate_results([])
        self.assertIsNone(result)