from django.core.management.base import BaseCommand
from doc_generator.utils import load_and_embed_documents

class Command(BaseCommand):
    help = (
        "Loads documents from the data/context directory, "
        "embeds them, and upserts them into the Chroma vector store."
    )

    def handle(self, *args, **options):
        self.stdout.write("Starting to build or update the vector store...")
        try:
            count = load_and_embed_documents()
            if count > 0:
                self.stdout.write(self.style.SUCCESS(f"Successfully processed {count} documents and updated the vector store."))
            else:
                self.stdout.write(self.style.WARNING("No new documents found to process."))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"An error occurred: {e}"))
            # For more detailed debugging, you might want to re-raise the exception
            # raise e