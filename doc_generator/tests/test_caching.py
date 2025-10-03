
from django.test import TestCase, override_settings
from django.core.cache import cache
from unittest.mock import MagicMock
import json

from doc_generator.services.retrieval_service import RetrievalService

# Use Django's in-memory cache backend for testing to avoid needing a live Redis server.
@override_settings(CACHES={'default': {'BACKEND': 'django.core.cache.backends.locmem.LocMemCache'}})
class CachingTests(TestCase):

    def setUp(self):
        # Clear the cache before each test
        cache.clear()
        self.mock_retriever = MagicMock()
        self.mock_llm = MagicMock()
        self.farm_data = {"catchment_name": "Test Catchment", "soil_type": "Silt Loam"}

    def test_llm_call_is_cached(self):
        """
        Tests that the result of an expensive LLM call (query generation) is cached
        and that the LLM is not called on the second request.
        """
        # 1. Configure the mock LLM to return a specific list of queries
        expected_queries = ["query 1", "query 2"]
        self.mock_llm.__or__.return_value.invoke.return_value = expected_queries

        # 2. Initialize the service and call the method for the first time
        service = RetrievalService(retriever=self.mock_retriever, llm=self.mock_llm)
        
        # First call - should be a cache MISS
        result1 = service.generate_query_variations(self.farm_data)

        # 3. Assertions for the first call
        self.assertEqual(result1, expected_queries)
        # The mock LLM chain should have been invoked once
        self.mock_llm.__or__.return_value.invoke.assert_called_once()

        # 4. Call the method for the second time with the same data
        # Second call - should be a cache HIT
        result2 = service.generate_query_variations(self.farm_data)

        # 5. Assertions for the second call
        self.assertEqual(result2, expected_queries)
        # The mock LLM chain should NOT have been invoked again. The call count remains 1.
        self.mock_llm.__or__.return_value.invoke.assert_called_once()

        # 6. Verify the item is in the cache
        farm_data_json = json.dumps(self.farm_data, sort_keys=True)
        cache_key = f"query_variations:{hash(farm_data_json)}"
        cached_value = cache.get(cache_key)
        self.assertEqual(cached_value, expected_queries)
