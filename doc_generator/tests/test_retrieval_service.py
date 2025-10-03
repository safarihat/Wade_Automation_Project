
from django.test import TestCase
from unittest.mock import MagicMock
from langchain_core.documents import Document

from doc_generator.services.retrieval_service import RetrievalService

class RetrievalServiceTests(TestCase):

    def setUp(self):
        # Mock the base retriever
        self.mock_retriever = MagicMock()
        # Mock the LLM for query generation
        self.mock_llm = MagicMock()

    def test_multi_query_deduplication(self):
        """
        Ensures that the multi_query_retrieve method correctly deduplicates
        documents retrieved from multiple queries.
        """
        # 1. Define the documents that will be "retrieved"
        doc1 = Document(page_content="This is the first document.", metadata={"source": "doc_a.pdf"})
        doc2 = Document(page_content="This is the second document.", metadata={"source": "doc_b.pdf"})
        doc3 = Document(page_content="This is a third, unique document.", metadata={"source": "doc_c.pdf"})

        # 2. Configure the mock retriever to return overlapping results
        self.mock_retriever.get_relevant_documents.side_effect = [
            [doc1, doc2],  # Results for query 1
            [doc2, doc3],  # Results for query 2
        ]

        # 3. Initialize and run the service
        service = RetrievalService(retriever=self.mock_retriever, llm=self.mock_llm)
        queries = ["query 1", "query 2"]
        final_docs = service.multi_query_retrieve(queries)

        # 4. Assertions
        self.assertEqual(self.mock_retriever.get_relevant_documents.call_count, 2)
        self.assertEqual(len(final_docs), 3)
        final_contents = {doc.page_content for doc in final_docs}
        self.assertIn("This is the first document.", final_contents)
        self.assertIn("This is the second document.", final_contents)
        self.assertIn("This is a third, unique document.", final_contents)

    def test_query_generation_fallback(self):
        """
        Tests that the query generation falls back to default queries if the LLM fails.
        """
        # 1. Configure the mock LLM to raise an exception
        self.mock_llm.__or__.return_value.invoke.side_effect = Exception("LLM is down")

        # 2. Initialize the service and generate queries
        service = RetrievalService(retriever=self.mock_retriever, llm=self.mock_llm)
        farm_data = {"catchment_name": "Test Catchment", "council_authority_name": "Test Council"}
        queries = service.generate_query_variations(farm_data)

        # 3. Assertions
        self.assertIn("Freshwater regulations in the Test Catchment catchment under the Southland Water and Land Plan", queries)
        self.assertIn("Good management practices for soil conservation in Test Council", queries)
