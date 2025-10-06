import logging
import json
from typing import List, Dict, Any, Tuple
from collections import defaultdict

from django.core.cache import cache
from django.conf import settings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser

logger = logging.getLogger(__name__)

class RetrievalService:
    """
    A service for advanced document retrieval that uses a multi-scope,
    weighted relevance model and a contextual back-fill mechanism.
    """

    def __init__(self, retriever: BaseRetriever, llm):
        """
        Initializes the service with specialized retrievers for each region scope.
        """
        self.llm = llm
        if not hasattr(retriever, 'vectorstore'):
            raise ValueError("The provided retriever must have a 'vectorstore' attribute.")
            
        self.vector_store = retriever.vectorstore
        self.catchment_retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"filter": {"region_scope": "catchment"}, "k": 8, "score_threshold": 0.5}
        )
        self.regional_retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"filter": {"region_scope": "regional"}, "k": 6, "score_threshold": 0.5}
        )
        self.national_retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"filter": {"region_scope": "national"}, "k": 4, "score_threshold": 0.5}
        )
        self.all_retrievers = {
            "catchment": self.catchment_retriever,
            "regional": self.regional_retriever,
            "national": self.national_retriever,
        }

    def generate_query_variations(self, farm_data: Dict[str, Any]) -> List[str]:
        """
        Generates a mix of LLM-based and static domain queries.
        """
        farm_data_json = json.dumps(farm_data, sort_keys=True)
        cache_key = f"query_variations_v3:{hash(farm_data_json)}" # v3 for new logic
        cached_queries = cache.get(cache_key)
        if cached_queries:
            logger.info(f"Cache HIT for query variations. Key: {cache_key}")
            return cached_queries

        logger.info(f"Cache MISS for query variations. Generating new queries.")
        catchment = farm_data.get("catchment_name", "the local")
        council = farm_data.get("council_authority_name", "regional council")
        variations = []
        
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

        domain_keywords = ["hydrology", "soil health", "erosion control", "groundwater quality", "riparian management", "land use policy"]
        domain_queries = [f"{keyword} in the {catchment} catchment" for keyword in domain_keywords]
        variations.extend(domain_queries)
        
        logger.info(f"Generated {len(variations)} total queries.")
        cache.set(cache_key, variations, timeout=settings.LLM_CACHE_TTL_SECONDS)
        return variations

    def _calculate_final_relevance(self, doc: Document, similarity_score: float) -> float:
        """
        Calculates the final relevance score based on similarity, scope, and semantic density.
        """
        metadata = doc.metadata
        scope = metadata.get("region_scope", "national")
        semantic_density = metadata.get("semantic_density", 0.5)

        scope_weights = {"catchment": 0.5, "regional": 0.35, "national": 0.15}
        scope_weight = scope_weights.get(scope, 0.15)

        # Final relevance formula from user request
        final_relevance = (
            (0.55 * similarity_score) +
            (0.25 * scope_weight) +
            (0.20 * semantic_density)
        )
        return final_relevance

    def multi_query_retrieve(self, queries: List[str]) -> List[Document]:
        """
        Executes multiple queries across different scopes, calculates a custom
        relevance score, merges the results, and applies contextual back-fill.
        """
        queries_json = json.dumps(sorted(queries), sort_keys=True)
        cache_key = f"multi_query_retrieve_v5:{hash(queries_json)}" # v5 for new logic
        cached_docs_data = cache.get(cache_key)
        if cached_docs_data:
            logger.info(f"Cache HIT for custom retrieval. Key: {cache_key}")
            return [Document(page_content=d['page_content'], metadata=d['metadata']) for d in cached_docs_data]

        logger.info(f"Cache MISS for custom retrieval. Performing retrieval...")
        
        # 1. Retrieve from all scopes
        all_scored_docs = []
        for scope, retriever in self.all_retrievers.items():
            for query in queries:
                try:
                    retrieved: List[Document] = retriever.invoke(query)
                    for doc in retrieved:
                        score = doc.metadata.get('score', 0.0)
                        all_scored_docs.append((doc, score))
                except Exception as e:
                    logger.error(f"Retrieval failed for query '{query}' in scope '{scope}': {e}")
                    continue
        
        # 2. Calculate final relevance and deduplicate
        unique_docs = {}
        for doc, score in all_scored_docs:
            final_score = self._calculate_final_relevance(doc, score)
            doc_id = (doc.metadata.get("source"), doc.page_content[:256])
            
            # Keep the document with the highest score
            if doc_id not in unique_docs or final_score > unique_docs[doc_id][1]:
                doc.metadata["final_relevance"] = final_score
                unique_docs[doc_id] = (doc, final_score)

        # 3. Sort by final relevance score
        sorted_docs = sorted(unique_docs.values(), key=lambda x: x[1], reverse=True)
        
        # Extract just the Document objects
        final_docs = [doc for doc, score in sorted_docs]

        # 4. Log retrieval balance
        catchment_count = sum(1 for d in final_docs if d.metadata.get("region_scope") == "catchment")
        regional_count = sum(1 for d in final_docs if d.metadata.get("region_scope") == "regional")
        national_count = sum(1 for d in final_docs if d.metadata.get("region_scope") == "national")
        logger.info(f"Retrieved document balance: {catchment_count} catchment, {regional_count} regional, {national_count} national.")

        # 5. Contextual Back-fill
        final_docs = self._fill_missing_themes(final_docs)

        serializable_docs = [{'page_content': d.page_content, 'metadata': d.metadata} for d in final_docs]
        cache.set(cache_key, serializable_docs, timeout=settings.LLM_CACHE_TTL_SECONDS)

        return final_docs

    def _fill_missing_themes(self, current_docs: List[Document]) -> List[Document]:
        """
        Analyzes the retrieved documents for thematic gaps and performs
        targeted searches to back-fill missing context.
        """
        themes = ["soil", "groundwater", "hydrology", "conservation", "management", "policy"]
        
        # Analyze current document set
        theme_counts = defaultdict(int)
        theme_density = defaultdict(list)
        for doc in current_docs:
            category = doc.metadata.get("category", "general")
            for theme in themes:
                if theme in category:
                    theme_counts[theme] += 1
                    theme_density[theme].append(doc.metadata.get("semantic_density", 0.0))

        # Identify and fill gaps
        docs_to_append = []
        for theme in themes:
            avg_density = sum(theme_density[theme]) / len(theme_density[theme]) if theme_density[theme] else 0
            
            if theme_counts[theme] == 0 or avg_density < 0.3:
                logger.info(f"Back-filling theme '{theme}' due to low coverage (count={theme_counts[theme]}, avg_density={avg_density:.2f}).")
                
                # Query for top 2 chunks for the missing theme
                backfill_retriever = self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={
                        "filter": {
                            "$and": [
                                {"region_scope": {"$in": ["regional", "national"]}},
                                {"category": "technical_data"}
                            ]
                        }, 
                        "k": 2
                    }
                )
                
                try:
                    # Use a generic query for the theme
                    backfill_docs: List[Document] = backfill_retriever.invoke(f"technical data about {theme}")
                    
                    for doc in backfill_docs:
                        doc.metadata["backfill"] = True
                        docs_to_append.append(doc)
                    
                    logger.info(f"Back-filled theme '{theme}' with {len(backfill_docs)} regional/national documents.")
                except Exception as e:
                    logger.error(f"Back-fill for theme '{theme}' failed: {e}")

        # Append back-filled docs and remove duplicates
        if docs_to_append:
            combined_docs = current_docs + docs_to_append
            unique_docs = {}
            for doc in combined_docs:
                doc_id = (doc.metadata.get("source"), doc.page_content[:256])
                if doc_id not in unique_docs:
                    unique_docs[doc_id] = doc
            return list(unique_docs.values())
            
        return current_docs