import sys
from pathlib import Path
from django.core.management.base import BaseCommand

# Add project root to sys.path to allow importing process_lawa_sites
# Assumes the command is in app/management/commands/ and the script is in the root.
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    from process_lawa_sites import main as process_lawa_sites_main
except ImportError:
    # Provide a helpful error message if the script is not found
    process_lawa_sites_main = None

class Command(BaseCommand):
    help = 'Updates the LAWA sites data by running the process_lawa_sites.py script.'

    def handle(self, *args, **options):
        if process_lawa_sites_main is None:
            self.stderr.write(self.style.ERROR(
                f"Could not import main function from process_lawa_sites.py. "
                f"Ensure the script exists in the project root: {project_root}"
            ))
            return

        self.stdout.write(self.style.SUCCESS('Starting LAWA site data update...'))
        try:
            process_lawa_sites_main()
            self.stdout.write(self.style.SUCCESS('Successfully updated LAWA site data.'))
        except Exception as e:
            self.stderr.write(self.style.ERROR(f'An error occurred during the update: {e}'))
