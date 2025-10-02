from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
import shutil
import os
import logging

from doc_generator.services.vector_store_builder import VectorStoreBuilder

logger = logging.getLogger(__name__)

VECTOR_STORE_PATH = getattr(settings, "VECTOR_STORE_PATH", "./vector_store")

class Command(BaseCommand):
    help = "Builds or updates the vector store using the default local pipeline."

    def add_arguments(self, parser):
        parser.add_argument(
            '--rebuild',
            action='store_true',
            help='Deletes the existing vector store and checkpoints before building.',
        )

    def handle(self, *args, **options):
        rebuild = options['rebuild']

        self.stdout.write(self.style.SUCCESS("Starting vector store build..."))

        if rebuild:
            self.stdout.write(self.style.WARNING(f"Rebuild requested: Deleting vector store at {VECTOR_STORE_PATH}..."))
            if os.path.exists(VECTOR_STORE_PATH):
                try:
                    shutil.rmtree(VECTOR_STORE_PATH)
                    self.stdout.write(self.style.SUCCESS("Successfully deleted old vector store."))
                except Exception as e:
                    raise CommandError(f"Error deleting vector store directory: {e}")
            else:
                self.stdout.write(self.style.NOTICE("Vector store directory not found, skipping deletion."))

        try:
            self.stdout.write("Executing local, resource-optimal pipeline.")
            builder = VectorStoreBuilder(rebuild=rebuild)
            builder.run()
            self.stdout.write(self.style.SUCCESS("Vector store build process finished."))

        except Exception as e:
            logger.exception("An unexpected error occurred during the build process:")
            raise CommandError(f"Build process failed: {e}")