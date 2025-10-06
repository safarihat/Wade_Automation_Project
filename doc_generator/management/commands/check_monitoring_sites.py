from django.core.management.base import BaseCommand
from doc_generator.models import MonitoringSite

class Command(BaseCommand):
    help = 'Checks the number of MonitoringSite objects in the database and prints the first site name'

    def handle(self, *args, **options):
        count = MonitoringSite.objects.count()
        self.stdout.write(f"Total MonitoringSites: {count}")
        if count > 0:
            first_site = MonitoringSite.objects.first()
            self.stdout.write(f"First site name: {first_site.site_name}")
        else:
            self.stdout.write("No monitoring sites found.")
