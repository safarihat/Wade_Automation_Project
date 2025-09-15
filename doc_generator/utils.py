import os
from django.conf import settings
import hashlib
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def load_and_embed_documents():
    # Path to the dedicated directory for RAG source documents
    data_dir = os.path.join(settings.BASE_DIR, 'doc_generator', 'data', 'context')
    
    # Load documents of different types. This pattern is easily extendable.
    all_documents = []

    # Load all .pdf files
    pdf_loader = DirectoryLoader(data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True)
    all_documents.extend(pdf_loader.load())

    # Load all .txt files
    txt_loader = DirectoryLoader(data_dir, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'}, show_progress=True)
    all_documents.extend(txt_loader.load())
    
    if not all_documents:
        print("Warning: No documents were loaded. Check the `rag_source_documents` directory and loader configurations.")
        return 0

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(all_documents)

    # Generate unique IDs for each chunk to enable incremental updates (upserts).
    # An ID is a hash of the source file and the content of the chunk.
    ids = [
        hashlib.md5(f"{doc.metadata['source']}-{doc.page_content}".encode()).hexdigest()
        for doc in texts
    ]

    # Define the embedding model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    # Explicitly set the device to 'cpu' to avoid "meta tensor" errors
    model_kwargs = {'device': 'cpu'}
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
    
    # Define the vector store path
    vector_store_path = os.path.join(settings.BASE_DIR, 'vector_store')

    # Create or load the vector store. By providing IDs, `add_documents` acts as an "upsert",
    # which is much more efficient than rebuilding the store from scratch.
    vector_store = Chroma(persist_directory=vector_store_path, embedding_function=embeddings)
    vector_store.add_documents(documents=texts, ids=ids)
    vector_store.persist()
    
    return len(all_documents)
