
import logging
import json
from typing import List, Dict, Any

from django.core.cache import cache
from django.conf import settings

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser

logger = logging.getLogger(__name__)

class RetrievalService:
    """
    A service for advanced document retrieval, including query transformation
    and multi-query retrieval with deduplication.
    """

    def __init__(self, retriever: BaseRetriever, llm):
        self.retriever = retriever
        self.llm = llm

    def generate_query_variations(self, farm_data: Dict[str, Any]) -> List[str]:
        """
        Uses an LLM to generate a list of 3-5 specific, technical questions
        to ask a regulatory database based on farm data. Caches the result.
        """
        # 1. Create a cache key based on the input farm data
        farm_data_json = json.dumps(farm_data, sort_keys=True)
        cache_key = f"query_variations:{hash(farm_data_json)}"

        # 2. Try to get the result from the cache
        cached_queries = cache.get(cache_key)
        if cached_queries:
            logger.info(f"Cache HIT for query variations. Key: {cache_key}")
            return cached_queries

        logger.info(f"Cache MISS for query variations. Generating new queries. Key: {cache_key}")
        output_parser = CommaSeparatedListOutputParser()
        template = """
        You are an expert environmental consultant in New Zealand. Based on the
        following summary of a farm's geospatial data, generate 3 to 5 concise,
        technical questions to ask a regulatory database.

        The questions should focus on identifying potential environmental risks,
        compliance obligations, and best practices related to the farm's specific
        characteristics, with a focus on the Southland Water and Land Plan.

        Farm Data Summary:
        {farm_data}

        {format_instructions}
        """
        prompt = PromptTemplate.from_template(
            template,
            partial_variables={"format_instructions": output_parser.get_format_instructions()}
        )
        chain = prompt | self.llm | output_parser

        try:
            variations = chain.invoke({"farm_data": farm_data})
            logger.info(f"Generated queries: {variations}")
            # 3. Set the result in the cache
            cache.set(cache_key, variations, timeout=settings.LLM_CACHE_TTL_SECONDS)
            return variations
        except Exception as e:
            logger.error(f"LLM-based query generation failed: {e}. Falling back to default queries.")
            catchment = farm_data.get("catchment_name", "New Zealand")
            council = farm_data.get("council_authority_name", "regional council")
            return [
                f"Freshwater regulations in the {catchment} catchment under the Southland Water and Land Plan",
                f"Good management practices for soil conservation in {council}",
                f"Nutrient management guidelines for {farm_data.get('soil_type', 'local')} soils in Southland",
            ]

    def multi_query_retrieve(self, queries: List[str]) -> List[Document]:
        """
        Executes multiple queries against the retriever and returns a single,
        deduplicated list of documents. Caches the result.
        """
        # 1. Create a cache key based on the sorted queries
        queries_json = json.dumps(sorted(queries), sort_keys=True)
        cache_key = f"multi_query_retrieve:{hash(queries_json)}"

        # 2. Try to get the result from the cache
        cached_docs_data = cache.get(cache_key)
        if cached_docs_data:
            logger.info(f"Cache HIT for multi-query retrieval. Key: {cache_key}")
            # Reconstruct Document objects from cached data
            return [Document(page_content=d['page_content'], metadata=d['metadata']) for d in cached_docs_data]

        logger.info(f"Cache MISS for multi-query retrieval. Performing retrieval. Key: {cache_key}")
        unique_docs = {}
        for query in queries:
            try:
                retrieved_docs: List[Document] = self.retriever.get_relevant_documents(query)
                for doc in retrieved_docs:
                    doc_id = (doc.metadata.get("source"), doc.page_content)
                    if doc_id not in unique_docs:
                        unique_docs[doc_id] = doc
            except Exception as e:
                logger.error(f"Retrieval failed for query '{query}': {e}")
                continue

        final_docs = list(unique_docs.values())
        logger.info(f"Retrieved {len(final_docs)} unique document chunks.")
        
        # 3. Cache the results before returning
        # Convert Document objects to a serializable format for caching
        serializable_docs = [{'page_content': d.page_content, 'metadata': d.metadata} for d in final_docs]
        cache.set(cache_key, serializable_docs, timeout=settings.LLM_CACHE_TTL_SECONDS)

        return final_docs
