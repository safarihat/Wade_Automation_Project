from django.core.management.base import BaseCommand, CommandError
from doc_generator.services.vector_store_builder import VectorStoreBuilder
from django.conf import settings
import chromadb
from collections import Counter
import os

class Command(BaseCommand):
    """
    A Django management command to build or update the vector store.
    """
    help = 'Builds or updates the document vector store and validates metadata.'

    def add_arguments(self, parser):
        parser.add_argument(
            '--rebuild',
            action='store_true',
            help='Force a full rebuild of the vector store, ignoring checkpoints.',
        )

    def handle(self, *args, **options):
        rebuild = options['rebuild']
        if rebuild:
            self.stdout.write(self.style.WARNING("--- Starting a full rebuild of the vector store. ---"))
        else:
            self.stdout.write(self.style.SUCCESS("--- Starting incremental update of the vector store... ---"))
        
        # Run the main build process
        VectorStoreBuilder(rebuild=rebuild).run()
        
        self.stdout.write(self.style.SUCCESS("--- Vector store operation complete. ---"))
        
        # --- Post-Build Validation Step ---
        self.stdout.write(self.style.HTTP_INFO("\n--- Post-Build Coverage Summary ---"))
        try:
            vector_store_path = str(settings.VECTOR_STORE_PATH)
            if not os.path.exists(vector_store_path):
                self.stdout.write(self.style.ERROR(f"Vector store path not found at '{vector_store_path}' for validation."))
                return

            client = chromadb.PersistentClient(path=vector_store_path)
            # Assumes the default collection name used by LangChain, which is typically "langchain"
            collection = client.get_collection(name="langchain") 
            
            metadata = collection.get(include=["metadatas"])
            
            if not metadata or not metadata['metadatas']:
                self.stdout.write(self.style.WARNING("Could not retrieve any metadata for validation."))
                return

            scope_counts = Counter(m.get('region_scope', 'undefined') for m in metadata['metadatas'])
            
            self.stdout.write(self.style.SUCCESS(f"Total vectors in store: {collection.count()}"))
            self.stdout.write("Document count by 'region_scope':")
            for scope, count in scope_counts.items():
                self.stdout.write(f"  - {scope}: {count} vectors")

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"An error occurred during post-build validation: {e}"))
            self.stdout.write(self.style.WARNING("Validation skipped. The store may be functional, but metadata could not be read."))