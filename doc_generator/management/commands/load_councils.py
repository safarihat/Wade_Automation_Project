import os
from django.core.management.base import BaseCommand
from django.contrib.gis.utils import LayerMapping
from django.conf import settings
from doc_generator.models import RegionalCouncil
from django.contrib.gis.gdal import DataSource

class Command(BaseCommand):
    help = 'Loads regional council boundaries from a shapefile into the database.'

    def add_arguments(self, parser):
        parser.add_argument(
            '--inspect',
            action='store_true',
            help='Inspect the shapefile and print its available fields without loading data.',
        )

    def handle(self, *args, **options):
        """
        Handles the execution of the management command.
        """
        # Path to the shapefile provided in the project
        shapefile_path = os.path.join(
            settings.BASE_DIR,
            'doc_generator',
            'data',
            'regional_council',
            'regional-council-2023-generalised.shp'
        )

        if not os.path.exists(shapefile_path):
            self.stdout.write(self.style.ERROR(f"Shapefile not found at: {shapefile_path}"))
            self.stdout.write(self.style.WARNING("Please ensure the shapefile and its related files (.shx, .dbf, etc.) are in the correct directory."))
            return

        # If the --inspect flag is used, print fields and exit.
        if options['inspect']:
            try:
                ds = DataSource(shapefile_path)
                layer = ds[0]
                # The `layer.fields` property directly returns a list of field name strings.
                # The previous list comprehension was incorrect.
                available_fields = layer.fields
                self.stdout.write(self.style.SUCCESS("--- Shapefile Field Inspection ---"))
                self.stdout.write(f"Available fields in '{os.path.basename(shapefile_path)}':")
                self.stdout.write(str(available_fields))
                self.stdout.write(self.style.NOTICE("\nUpdate the 'name' key in the 'council_mapping' dictionary in this script with the correct field name, then run the command again without '--inspect'."))
                return
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Could not inspect shapefile: {e}"))
                return

        # Mapping dictionary from the RegionalCouncil model fields to the shapefile's fields.
        # The original field name 'REGC2023_V1_00_NAME' is truncated due to the 10-character
        # limit in shapefile (.dbf) field names. Inspection reveals the correct field is 'REGC2023_1'.
        council_mapping = {
            'name': 'REGC2023_1',
            'geom': 'MULTIPOLYGON',
        }

        # The LayerMapping utility handles the data loading and projection transformation.
        # `transform=True` converts the data from its source projection (NZTM, SRID 2193)
        # to the model's required projection (WGS84, SRID 4326).
        # We explicitly set `source_srid=2193` for robustness.
        lm = LayerMapping(
            RegionalCouncil,
            shapefile_path,
            council_mapping,
            source_srs=2193,
            transform=True,
            encoding='utf-8'
        )

        # Clear existing data to ensure a clean import with the correct projection.
        self.stdout.write("Clearing old regional council data...")
        RegionalCouncil.objects.all().delete()

        # Load the new data.
        self.stdout.write("Loading new regional council data with correct projection...")
        lm.save(strict=True, verbose=True)

        self.stdout.write(self.style.SUCCESS(f"Successfully loaded {RegionalCouncil.objects.count()} regional councils."))