
import os
import json
import time
import hashlib
import torch
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional

from django.conf import settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from unstructured.partition.pdf import partition_pdf

# --- Configuration ---
# Sensible defaults for low-resource machines, configurable via Django settings.
# Resource Optimization: Allows tuning the process to match machine specs.
DOCS_PATH = getattr(settings, "DOCS_PATH", "./doc_generator/data/context")
VECTOR_STORE_PATH = getattr(settings, "VECTOR_STORE_PATH", "./vector_store")
CHECKPOINT_FILE = os.path.join(VECTOR_STORE_PATH, "checkpoint.json")

# The embedding model is now definitively set in settings.py.
# We use getattr without a default to ensure the build fails loudly
# if this critical setting is missing, preventing silent errors.
EMBEDDING_MODEL_NAME = getattr(settings, "VECTOR_STORE_EMBEDDING_MODEL_NAME")

# Resource Optimization: Limits how many large files are processed at once.
MAX_WORKERS = getattr(settings, "VECTOR_STORE_MAX_WORKERS", 2)

# Resource Optimization: Small batch size to avoid memory overflow during embedding.
EMBEDDING_BATCH_SIZE = getattr(settings, "VECTOR_STORE_EMBEDDING_BATCH_SIZE", 16)

CHUNK_SIZE = getattr(settings, "VECTOR_STORE_CHUNK_SIZE", 1000)
CHUNK_OVERLAP = getattr(settings, "VECTOR_STORE_CHUNK_OVERLAP", 200)

class VectorStoreBuilder:
    """A robust, resource-optimal service for building a vector store."""

    def __init__(self, rebuild=False):
        self.rebuild = rebuild
        self.device = self._get_device()
        self.embedding_model = self._init_embedding_model()
        self.vector_store = self._init_vector_store()
        self.checkpoints = self._load_checkpoints()

    def _get_device(self) -> str:
        """Detects and returns the best available device for torch."""
        if torch.cuda.is_available():
            print("GPU (CUDA) detected. Using for embeddings.")
            return "cuda"
        # MPS support can be unstable. Add check if needed.
        # if torch.backends.mps.is_available():
        #     print("GPU (MPS) detected. Using for embeddings.")
        #     return "mps"
        print("No GPU detected. Using CPU for embeddings.")
        return "cpu"

    def _init_embedding_model(self) -> HuggingFaceEmbeddings:
        """Initializes the HuggingFace embedding model."""
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": self.device},
        )

    def _init_vector_store(self) -> Chroma:
        """Initializes the Chroma vector store client."""
        return Chroma(
            persist_directory=VECTOR_STORE_PATH,
            embedding_function=self.embedding_model,
        )

    def _load_checkpoints(self) -> Dict[str, str]:
        """Loads the checkpoint file if it exists."""
        if self.rebuild:
            print("Rebuild requested. Ignoring checkpoints.")
            return {}
        try:
            if os.path.exists(CHECKPOINT_FILE):
                with open(CHECKPOINT_FILE, "r") as f:
                    print(f"Checkpoint file found. Loading progress from {CHECKPOINT_FILE}")
                    return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load checkpoint file. Starting fresh. Error: {e}")
        return {}

    def _save_checkpoint(self, file_path: str, file_hash: str):
        """Saves the hash of a processed file to the checkpoint file."""
        self.checkpoints[file_path] = file_hash
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump(self.checkpoints, f, indent=4)

    def _get_file_hash(self, file_path: str) -> str:
        """Computes the SHA256 hash of a file's content."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()

    def run(self):
        """Main entry point to run the vector store build process."""
        print(f"Starting vector store build with max_workers={MAX_WORKERS}...")
        start_time = time.time()

        all_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(DOCS_PATH) for f in filenames if f.endswith(".pdf")]
        
        files_to_process = self._filter_files_for_processing(all_files)

        if not files_to_process:
            print("Vector store is already up-to-date.")
            return

        # Resource Optimization: ThreadPoolExecutor limits concurrent file processing.
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            executor.map(self._process_file, files_to_process)

        # Efficiency Optimization: Persist is called only once after all files are processed.
        print("All files processed. Persisting vector store to disk...")
        self.vector_store.persist()
        
        total_time = time.time() - start_time
        print(f"Vector store build completed in {total_time:.2f} seconds.")

    def _filter_files_for_processing(self, all_files: List[str]) -> List[str]:
        """Determines which files need to be processed based on checkpoints."""
        if self.rebuild:
            return all_files

        files_to_process = []
        for file_path in all_files:
            file_hash = self._get_file_hash(file_path)
            if self.checkpoints.get(file_path) != file_hash:
                files_to_process.append(file_path)
            else:
                print(f"Skipping already processed file: {os.path.basename(file_path)}")
        
        print(f"Found {len(files_to_process)} new or modified files to process.")
        return files_to_process

    def _process_file(self, file_path: str):
        """The complete processing pipeline for a single file."""
        try:
            file_name = os.path.basename(file_path)
            print(f"Processing file: {file_name}")
            file_hash = self._get_file_hash(file_path)

            # 1. Parse PDF
            # Resource Optimization: `strategy="fast"` is much lighter than `hi_res`.
            elements = partition_pdf(
                file_path,
                strategy="fast",
                languages=["eng", "mri"] # Support for English and Māori
            )
            
            # 2. Chunk Elements
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            texts = [el.text for el in elements]
            
            # Extract metadata including region
            base_metadata = {"source": file_name}
            region = self._extract_region_from_path(file_path)
            if region:
                base_metadata["region"] = region
            
            chunks = text_splitter.create_documents(texts, metadatas=[base_metadata] * len(texts))

            # 3. Embed and Add to Store in Batches
            # Resource Optimization: Processing in small batches prevents memory spikes.
            total_chunks = len(chunks)
            for i in range(0, total_chunks, EMBEDDING_BATCH_SIZE):
                batch = chunks[i:i + EMBEDDING_BATCH_SIZE]
                print(f"  - Embedding batch {i//EMBEDDING_BATCH_SIZE + 1}/{(total_chunks + EMBEDDING_BATCH_SIZE - 1)//EMBEDDING_BATCH_SIZE} for {file_name}")
                self.vector_store.add_documents(documents=batch)

            # 4. Checkpoint
            # Resiliency: If the process crashes, we won't have to re-process this file.
            self._save_checkpoint(file_path, file_hash)
            print(f"Finished processing and checkpointed file: {file_name}")

        except Exception as e:
            print(f"ERROR: Failed to process file {file_path}. Reason: {e}")
            # Optionally, log this to a separate error file.

    def _extract_region_from_path(self, file_path: str) -> Optional[str]:
        """Extracts the region from the file path based on the DOCS_PATH structure."""
        try:
            relative_path = os.path.relpath(file_path, DOCS_PATH)
            # Assuming DOCS_PATH is like 'doc_generator/data/context'
            # and region is the first subdirectory, e.g., 'southland/file.pdf'
            parts = relative_path.split(os.sep)
            if len(parts) > 1 and parts[0] != os.path.basename(file_path): # Ensure it's a subdirectory, not just a file in DOCS_PATH
                return parts[0]
        except ValueError:
            pass # file_path is not in DOCS_PATH
        return None
