from django.conf import settings
from langchain_huggingface import HuggingFaceEmbeddings

# This variable will hold the singleton instance of the embedding model.
_embedding_model = None

def get_embedding_model():
    """
    Initializes and returns a singleton instance of the embedding model.
    This ensures the model is loaded into memory only once.
    """
    global _embedding_model
    if _embedding_model is None:
        model_name = settings.VECTOR_STORE_EMBEDDING_MODEL_NAME
        # Explicitly set the device to 'cpu' to avoid potential issues on servers
        # without a GPU or with conflicting CUDA/torch versions.
        model_kwargs = {'device': 'cpu'}
        _embedding_model = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
    return _embedding_model
