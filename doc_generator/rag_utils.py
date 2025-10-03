
import os
import hashlib
import logging

from django.conf import settings
from langchain.docstore.document import Document
from langchain_chroma import Chroma

from .pdf_utils import process_pdf_with_unstructured
from .services.embedding_service import get_embedding_model

logger = logging.getLogger(__name__)

def load_and_embed_documents():
    """
    Loads PDF documents, processes them, embeds them, and upserts into Chroma.
    """
    data_dir = os.path.join(settings.BASE_DIR, 'doc_generator', 'data', 'context')
    pdf_files = [os.path.join(root, f) for root, _, files in os.walk(data_dir) for f in files if f.endswith(".pdf")]

    if not pdf_files:
        logger.warning(f"No PDF documents found in {data_dir}.")
        return 0

    all_docs = []
    for pdf_file in pdf_files:
        logger.info(f"Processing: {pdf_file}")
        all_docs.extend(process_pdf_with_unstructured(pdf_file))

    if not all_docs:
        logger.warning("No processable content found in the documents.")
        return 0

    ids = [hashlib.md5(f"{doc.metadata['document_source']}-{doc.metadata['page_number']}-{doc.page_content}".encode()).hexdigest() for doc in all_docs]

    embeddings = get_embedding_model()
    vector_store_path = os.path.join(settings.BASE_DIR, 'vector_store')
    vector_store = Chroma(persist_directory=vector_store_path, embedding_function=embeddings)

    logger.info(f"Upserting {len(all_docs)} chunks into Chroma vector store.")
    vector_store.add_documents(documents=all_docs, ids=ids)
    vector_store.persist()

    logger.info("Vector store update complete.")
    return len(all_docs)
