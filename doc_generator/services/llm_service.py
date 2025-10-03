import json
import logging
from typing import List, Dict, Any

# LangChain components
from langchain_core.documents import Document
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_groq import ChatGroq

# Configure a logger for this module
logger = logging.getLogger(__name__)


class LLMService:
    """
    A service class for handling interactions with a Large Language Model (LLM)
    for generating content related to freshwater farm plans.
    """

    @staticmethod
    def generate_risk_matrix(plan):
        """
        Generates a risk matrix for a given freshwater plan.

        For now, this method returns a stubbed JSON structure.
        Later, this will be extended to call a RAG model to pre-populate
        inherent vulnerabilities, catchment context, and potential risks.

        Args:
            plan (FreshwaterPlan): The freshwater plan instance.

        Returns:
            str: A JSON string representing the initial risk table data.
        """
        # If the plan already has risk data, return it. Otherwise, return placeholder data.
        if plan.risk_management_data:
            return json.dumps(plan.risk_management_data)

        placeholder_data = [
            {
                "land_unit": "1",
                "activity_group": "Stock exclusion",
                "sub_group": "Bridges or culverts",
                "activity_description": "Placeholder activity description.",
                "inherent_vulnerabilities": "Placeholder vulnerability context from Step 3/4.",
                "catchment_context": "Placeholder catchment context from RAG.",
                "risk_group": "Water quality",
                "risk": "Placeholder risk assessment from LLM."
            }
        ]
        return json.dumps(placeholder_data)

    @staticmethod
    def _simplify_geospatial_data_for_prompt(farm_data: Dict[str, Any]) -> str:
        """
        Simplifies complex geospatial data into a concise, human-readable
        JSON string suitable for an LLM prompt. It extracts key properties
        from nested dictionaries.
        """
        summary = {
            'catchment_name': farm_data.get('catchment_name'),
            'council_authority_name': farm_data.get('council_authority_name'),
            'slope_degrees': farm_data.get('arcgis_slope_angle'),
            'nutrient_leaching_vulnerability': farm_data.get('nutrient_leaching_vulnerability'),
            'erodibility': farm_data.get('erodibility'),
            'soil_type': farm_data.get('soil_type'),
        }

        # Extract nested regional soil data if available
        if farm_data.get("regional_soil", {}).get("features"):
            props = farm_data["regional_soil"]["features"][0].get("properties", {})
            summary['soil_drainage'] = props.get('Darg_Drain')
            summary['soil_rock_type'] = props.get('Darg_Rock')

        # Clean the summary by removing any keys with None values
        simplified_data = {k: v for k, v in summary.items() if v is not None}
        return json.dumps(simplified_data, indent=2)


    @staticmethod
    def _generate_queries_from_farm_data(farm_data_summary: str) -> List[str]:
        """
        Uses an LLM to generate specific, targeted search queries based on farm data.

        This function prompts the model to act as an environmental consultant and return
        a structured list of questions for a regulatory database search. It uses
        tool-calling features to ensure a reliable, structured output.

        Args:
            farm_data_summary: A JSON string summarizing the farm's key geospatial
                               and environmental characteristics.

        Returns:
            A list of generated query strings.
        """
        from langchain_core.tools import tool
        from langchain_core.pydantic_v1 import BaseModel, Field

        @tool
        def regulatory_query_generator(queries: List[str]):
            """Generates a list of regulatory search queries."""
            pass

        class RegulatoryQueryGenerator(BaseModel):
            queries: List[str] = Field(
                ...,
                description=(
                    "A list of 3-5 specific and technical questions to ask a "
                    "New Zealand environmental regulatory database."
                ),
            )

        llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
        llm_with_tools = llm.with_structured_output(RegulatoryQueryGenerator)

        prompt = PromptTemplate.from_template(
            """
            You are an expert environmental consultant in New Zealand. Based on the
            following summary of a farm's geospatial data, generate 3 to 5 concise,
            technical questions to ask a regulatory database.

            The questions should focus on identifying potential environmental risks,
            compliance obligations, and best practices related to the farm's specific
            characteristics.

            Farm Data Summary:
            {farm_data_summary}
            """
        )

        chain = prompt | llm_with_tools
        response = chain.invoke({"farm_data_summary": farm_data_summary})
        return response.queries


    @staticmethod
    def generate_retrieval_context(farm_data: Dict[str, Any], retriever: BaseRetriever) -> List[str]:
        """
        Generates and executes multiple targeted queries against a vector store to
        retrieve relevant regulatory context based on geospatial data.

        This function orchestrates the query generation and retrieval process. It first
        generates queries using an LLM, then executes each query against the provided
        retriever, and finally combines the unique results into a single context.

        Args:
            farm_data: A dictionary containing the farm's data.
            retriever: A LangChain retriever instance (e.g., from Chroma).

        Returns:
            A list of unique, relevant document contents for the RAG context.
        """
        generated_queries = []
        try:
            farm_data_summary = LLMService._simplify_geospatial_data_for_prompt(farm_data)
            generated_queries = LLMService._generate_queries_from_farm_data(farm_data_summary)
            logger.info(f"Generated retrieval queries: {generated_queries}")
        except Exception as e:
            logger.warning(
                f"LLM-based query generation failed: {e}. "
                f"Falling back to a set of default descriptive queries."
            )

        # Fallback mechanism if LLM query generation fails or returns no queries
        if not generated_queries:
            catchment = farm_data.get("catchment_name", "New Zealand")
            council = farm_data.get("council_authority_name", "regional council")
            generated_queries = [
                f"Freshwater regulations in the {catchment} catchment",
                f"Good management practices for soil conservation in {council}",
                f"Nutrient management guidelines for {farm_data.get('soil_type', 'local')} soils",
                f"Erosion control on steep slopes in New Zealand",
                f"Water quality rules for {council}",
            ]

        # Retrieve and combine unique documents
        unique_docs = {}
        for query in generated_queries:
            try:
                retrieved_docs: List[Document] = retriever.invoke(query)
                for doc in retrieved_docs:
                    # Use a combination of source and content for a more robust uniqueness check
                    doc_id = (doc.metadata.get("source"), doc.page_content)
                    if doc_id not in unique_docs:
                        unique_docs[doc_id] = doc
            except Exception as e:
                logger.error(f"Retrieval failed for query '{query}': {e}")
                continue

        final_context = [doc.page_content for doc in unique_docs.values()]
        logger.info(f"Retrieved {len(final_context)} unique document chunks for context.")
        return final_context
