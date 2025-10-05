from django.core.management.base import BaseCommand
import json
import os
from django.conf import settings
from collections import Counter

class Command(BaseCommand):
    help = "Analyzes retrieval diagnostic logs to summarize category balance and highlight underrepresented topics."

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS("Analyzing retrieval logs..."))
        
        log_dir = settings.BASE_DIR / 'logs'
        report_path = log_dir / 'retrieval_balance_report.json'

        if not os.path.exists(report_path):
            self.stdout.write(self.style.ERROR(f"Report file not found at {report_path}"))
            self.stdout.write(self.style.WARNING("Please run a vulnerability analysis first to generate a report."))
            return

        with open(report_path, 'r', encoding='utf-8') as f:
            report_data = json.load(f)

        category_balance = report_data.get("category_balance", {})
        
        if not category_balance:
            self.stdout.write(self.style.WARNING("No category information found in the report."))
            return

        self.stdout.write(self.style.SUCCESS("\n--- Retrieval Balance Report ---"))
        self.stdout.write(f"Report generated at: {report_data.get('report_generated_at')}")
        self.stdout.write(f"Total documents retrieved: {report_data.get('total_docs_retrieved')}\n")

        self.stdout.write(self.style.HTTP_INFO("Category Distribution:"))
        for category, count in category_balance.items():
            self.stdout.write(f"  - {category}: {count} docs")

        # Identify underrepresented topics
        underrepresented = [cat for cat, count in category_balance.items() if count < 2] # Example threshold
        if underrepresented:
            self.stdout.write(self.style.WARNING("\nPotential underrepresented topics (fewer than 2 docs):"))
            for cat in underrepresented:
                self.stdout.write(f"  - {cat}")
        else:
            self.stdout.write(self.style.SUCCESS("\nCategory representation appears balanced."))
            
        self.stdout.write("\n--- End of Report ---\n")