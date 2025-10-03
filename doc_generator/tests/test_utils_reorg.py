
from django.test import TestCase
from unittest.mock import patch

class UtilsReorganizationTests(TestCase):

    def test_geospatial_imports(self):
        """Smoke test to ensure geospatial functions are importable from their new location."""
        try:
            from doc_generator.geospatial_utils import transform_coords, _query_arcgis_vector
            self.assertTrue(callable(transform_coords))
            self.assertTrue(callable(_query_arcgis_vector))
        except ImportError as e:
            self.fail(f"Failed to import geospatial utils: {e}")

    def test_pdf_imports(self):
        """Smoke test to ensure PDF functions are importable from their new location."""
        try:
            from doc_generator.pdf_utils import process_pdf_with_unstructured
            self.assertTrue(callable(process_pdf_with_unstructured))
        except ImportError as e:
            self.fail(f"Failed to import pdf utils: {e}")

    def test_rag_imports(self):
        """Smoke test to ensure RAG functions are importable from their new location."""
        try:
            from doc_generator.rag_utils import load_and_embed_documents
            self.assertTrue(callable(load_and_embed_documents))
        except ImportError as e:
            self.fail(f"Failed to import rag utils: {e}")
