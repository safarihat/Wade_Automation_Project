from django.core.management.base import BaseCommand
from doc_generator.utils import load_and_embed_documents

class Command(BaseCommand):
    help = 'Builds the vector store for the RAG pipeline.'

    def handle(self, *args, **options):
        self.stdout.write("Building vector store...")
        num_documents = load_and_embed_documents()
        self.stdout.write(self.style.SUCCESS(f'Successfully built vector store with {num_documents} documents.'))
