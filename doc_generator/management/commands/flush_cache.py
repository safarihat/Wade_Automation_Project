from django.core.management.base import BaseCommand
from django.core.cache import cache

class Command(BaseCommand):
    help = 'Flushes the Redis cache.'

    def handle(self, *args, **options):
        self.stdout.write('Flushing Redis cache...')
        cache.clear()
        self.stdout.write(self.style.SUCCESS('Successfully flushed Redis cache.'))
