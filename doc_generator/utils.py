import os
from django.conf import settings
import hashlib
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def load_and_embed_documents():
    # Construct the path to the data directory
    data_dir = os.path.join(settings.BASE_DIR, 'doc_generator', 'data', 'regional_council')
    
    # Load documents
    loader = DirectoryLoader(data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # Generate unique IDs for each chunk to enable incremental updates (upserts).
    # An ID is a hash of the source file and the content of the chunk.
    ids = [
        hashlib.md5(f"{doc.metadata['source']}-{doc.page_content}".encode()).hexdigest()
        for doc in texts
    ]

    # Define the embedding model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    # Define the vector store path
    vector_store_path = os.path.join(settings.BASE_DIR, 'vector_store')

    # Create or load the vector store. By providing IDs, `add_documents` acts as an "upsert",
    # which is much more efficient than rebuilding the store from scratch.
    vector_store = Chroma(persist_directory=vector_store_path, embedding_function=embeddings)
    vector_store.add_documents(documents=texts, ids=ids)
    vector_store.persist()
    
    return len(documents)
