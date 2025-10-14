# doc_generator/services/rag_service.py
import os
import logging
from django.conf import settings

# LangChain components
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from .embedding_service import get_embedding_model

logger = logging.getLogger(__name__)

class RAGService:
    """
    A service for performing Retrieval-Augmented Generation (RAG) tasks.
    """
    def __init__(self):
        """Initializes the RAG service with the necessary components."""
        try:
            embeddings = get_embedding_model()
            vector_store_path = os.path.join(settings.BASE_DIR, 'vector_store')
            self.vector_store = Chroma(persist_directory=vector_store_path, embedding_function=embeddings)
            self.retriever = self.vector_store.as_retriever(search_kwargs={'k': 3})
            self.llm = ChatGroq(model_name=settings.GROQ_MODEL, groq_api_key=settings.GROQ_API_KEY)
        except Exception as e:
            logger.error(f"Failed to initialize RAGService: {e}", exc_info=True)
            raise

    def query_council_authority(self, council_name: str) -> str:
        """
        Uses RAG to find the official name for a given council.
        """
        template = """
        Based on the provided context about New Zealand regional councils, what is the official full name for the council referred to as '{council_name}'?
        Provide only the official name and nothing else.

        Context:
        {context}
        """
        prompt = PromptTemplate.from_template(template)

        rag_chain = (
            {"context": self.retriever, "council_name": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        verified_name = rag_chain.invoke(council_name)
        return verified_name.strip()