import logging
import json
from typing import List, Dict, Any

from django.core.cache import cache
from django.conf import settings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain.retrievers import EnsembleRetriever

logger = logging.getLogger(__name__)

class RetrievalService:
    """
    A service for advanced document retrieval that combines results from
    catchment-specific and regional-technical documents.
    """

    def __init__(self, retriever: BaseRetriever, llm):
        """
        Initializes the service with an ensemble retriever that balances
        catchment and regional documents.
        """
        self.llm = llm
        
        # The base_retriever is expected to be a VectorStoreRetriever.
        # We use its underlying vectorstore to create specialized retrievers.
        if not hasattr(retriever, 'vectorstore'):
            raise ValueError("The provided retriever must have a 'vectorstore' attribute.")
            
        vector_store = retriever.vectorstore

        # 1. Create a retriever for catchment-specific documents
        catchment_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"filter": {"region_scope": "catchment"}, "k": 5}
        )

        # 2. Create a retriever for regional technical documents
        regional_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"filter": {"region_scope": "regional"}, "k": 3}
        )

        # 3. Create the ensemble with the requested 70/30 weighting
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[catchment_retriever, regional_retriever],
            weights=[0.7, 0.3]
        )

    def generate_query_variations(self, farm_data: Dict[str, Any]) -> List[str]:
        """
        Generates a mix of LLM-based and static domain queries.
        """
        farm_data_json = json.dumps(farm_data, sort_keys=True)
        cache_key = f"query_variations_v2:{hash(farm_data_json)}"
        cached_queries = cache.get(cache_key)
        if cached_queries:
            logger.info(f"Cache HIT for query variations. Key: {cache_key}")
            return cached_queries

        logger.info(f"Cache MISS for query variations. Generating new queries.")
        catchment = farm_data.get("catchment_name", "the local")
        council = farm_data.get("council_authority_name", "regional council")
        variations = []
        
        # LLM-generated queries
        output_parser = CommaSeparatedListOutputParser()
        template = """
        Based on the farm's geospatial data, generate 3-5 concise, technical questions
        to ask a regulatory and environmental database. Focus on risks and obligations.
        Farm Data Summary: {farm_data}
        {format_instructions}
        """
        prompt = PromptTemplate.from_template(
            template,
            partial_variables={"format_instructions": output_parser.get_format_instructions()}
        )
        chain = prompt | self.llm | output_parser
        try:
            llm_variations = chain.invoke({"farm_data": farm_data})
            variations.extend(llm_variations)
        except Exception as e:
            logger.error(f"LLM-based query generation failed: {e}. Using fallbacks.")
            variations.extend([
                f"Freshwater regulations in the {catchment} catchment",
                f"Soil conservation practices for {council}",
            ])

        # Domain-specific queries
        domain_keywords = ["hydrology", "soil health", "erosion control", "groundwater quality", "riparian management"]
        domain_queries = [f"{keyword} in the {catchment} catchment" for keyword in domain_keywords]
        variations.extend(domain_queries)
        
        logger.info(f"Generated {len(variations)} total queries.")
        cache.set(cache_key, variations, timeout=settings.LLM_CACHE_TTL_SECONDS)
        return variations

    def multi_query_retrieve(self, queries: List[str]) -> List[Document]:
        """
        Executes multiple queries using the ensemble retriever and returns a
        single, deduplicated list of documents.
        """
        queries_json = json.dumps(sorted(queries), sort_keys=True)
        cache_key = f"multi_query_retrieve_v4:{hash(queries_json)}" # v4 for new logic
        cached_docs_data = cache.get(cache_key)
        if cached_docs_data:
            logger.info(f"Cache HIT for ensemble retrieval. Key: {cache_key}")
            return [Document(page_content=d['page_content'], metadata=d['metadata']) for d in cached_docs_data]

        logger.info(f"Cache MISS for ensemble retrieval. Performing retrieval...")
        
        # The ensemble retriever handles the complexity of fetching and merging.
        # We collect results from all queries.
        unique_docs = {}
        for query in queries:
            try:
                retrieved_docs: List[Document] = self.ensemble_retriever.invoke(query)
                for doc in retrieved_docs:
                    # Use a tuple of source and a content snippet as a unique key
                    doc_id = (doc.metadata.get("source"), doc.page_content[:256])
                    if doc_id not in unique_docs:
                        unique_docs[doc_id] = doc
            except Exception as e:
                logger.error(f"Retrieval failed for query '{query}': {e}")
                continue

        final_docs = list(unique_docs.values())
        logger.info(f"Retrieved {len(final_docs)} unique document chunks from ensemble.")
        
        # Log the balance for verification
        catchment_count = sum(1 for d in final_docs if d.metadata.get("region_scope") == "catchment")
        regional_count = sum(1 for d in final_docs if d.metadata.get("region_scope") == "regional")
        logger.info(f"Retrieved document balance: {catchment_count} catchment, {regional_count} regional.")

        serializable_docs = [{'page_content': d.page_content, 'metadata': d.metadata} for d in final_docs]
        cache.set(cache_key, serializable_docs, timeout=settings.LLM_CACHE_TTL_SECONDS)

        return final_docs
