
from django.core.management.base import BaseCommand
from doc_generator.models import MonitoringSite

class Command(BaseCommand):
    help = 'Prints the content of the MonitoringSite table.'

    def handle(self, *args, **options):
        self.stdout.write("Monitoring Sites in the database:")
        sites = MonitoringSite.objects.all()
        if sites:
            for site in sites:
                self.stdout.write(f"- {site.site_name} ({site.location.x}, {site.location.y})")
        else:
            self.stdout.write("No sites found.")
