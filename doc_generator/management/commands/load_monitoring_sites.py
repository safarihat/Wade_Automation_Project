import requests
import xml.etree.ElementTree as ET
import csv
from django.core.management.base import BaseCommand
from django.contrib.gis.geos import Point
from doc_generator.models import MonitoringSite

class Command(BaseCommand):
    help = 'Loads water quality monitoring sites from Environment Southland\'s Hilltop API.'

    HILLTOP_URL = "http://odp.es.govt.nz/data.hts?Service=Hilltop&Request=SiteList"
    RELEVANT_MEASUREMENTS = {
        "E-Coli <CFU>",
        "Nitrogen (Nitrate Nitrite)",
        "Phosphorus (Dissolved Reactive)",
        "Turbidity (Lab)",
        "Turbidity (FNU)",
    }

    def handle(self, *args, **options):
        self.stdout.write("Fetching monitoring sites from Hilltop API...")

        try:
            response = requests.get(self.HILLTOP_URL)
            response.raise_for_status()
            root = ET.fromstring(response.content)
        except (requests.exceptions.RequestException, ET.ParseError) as e:
            self.stderr.write(self.style.ERROR(f"Failed to fetch or parse site list: {e}"))
            # Fallback to CSV if API fails
            self.load_from_csv()
            return

        sites_created = 0
        sites_updated = 0

        for site_node in root.findall('Site'):
            site_name = site_node.get('Name')
            latitude = site_node.get('Latitude')
            longitude = site_node.get('Longitude')

            if not all([site_name, latitude, longitude]):
                continue

            # Check if the site has at least one relevant measurement
            has_relevant_measurement = False
            for measurement_node in site_node.findall('.//Measurement'):
                measurement_name = measurement_node.get('Name')
                if measurement_name in self.RELEVANT_MEASUREMENTS:
                    has_relevant_measurement = True
                    break
            
            if not has_relevant_measurement:
                continue

            try:
                location = Point(float(longitude), float(latitude), srid=4326)
            except (ValueError, TypeError):
                self.stdout.write(self.style.WARNING(f"Could not parse coordinates for site: {site_name}"))
                continue

            obj, created = MonitoringSite.objects.update_or_create(
                hilltop_site_id=site_name,
                defaults={
                    'site_name': site_name,
                    'location': location
                }
            )

            if created:
                sites_created += 1
            else:
                sites_updated += 1

        if sites_created == 0 and sites_updated == 0:
            self.stdout.write(self.style.WARNING("No relevant sites found from Hilltop API. Falling back to CSV."))
            self.load_from_csv()
        else:
            self.stdout.write(self.style.SUCCESS(
                f"Successfully processed sites from Hilltop API. Created: {sites_created}, Updated: {sites_updated}."
            ))

    def load_from_csv(self):
        self.stdout.write("Loading monitoring sites from monitoring_sites.csv...")
        try:
            with open('monitoring_sites.csv', 'r') as f:
                reader = csv.DictReader(f)
                sites = list(reader)
        except FileNotFoundError:
            self.stderr.write(self.style.ERROR("monitoring_sites.csv not found. Please create this file."))
            return

        sites_created = 0
        for site in sites:
            site_name = site.get('site_name')
            latitude = site.get('latitude')
            longitude = site.get('longitude')
            if not all([site_name, latitude, longitude]):
                continue
            try:
                location = Point(float(longitude), float(latitude), srid=4326)
                MonitoringSite.objects.get_or_create(
                    hilltop_site_id=site_name,
                    defaults={'site_name': site_name, 'location': location}
                )
                sites_created += 1
            except (ValueError, TypeError):
                continue
        self.stdout.write(self.style.SUCCESS(f"Created {sites_created} sites from CSV."))