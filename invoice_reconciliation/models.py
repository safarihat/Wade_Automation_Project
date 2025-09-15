from django.db import models
from django.contrib.auth.models import User
from decimal import Decimal

class InvoiceManager(models.Manager):
    def create_with_lines(self, user, invoice_data: dict, lines_text: str):
        """
        Creates an Invoice and its associated InvoiceLine objects from structured data.
        """
        invoice = self.create(user=user, **invoice_data)

        order = 1
        for row in lines_text.splitlines():
            row = row.strip()
            if not row:
                continue
            parts = row.split('|') + [''] * 4 # Ensure at least 4 parts
            
            InvoiceLine.objects.create(
                invoice=invoice,
                description=parts[0].strip()[:512],
                quantity=Decimal(parts[1]) if parts[1] else None,
                unit_price=Decimal(parts[2]) if parts[2] else None,
                line_total=Decimal(parts[3]) if parts[3] else None,
                order=order
            )
            order += 1
        return invoice

class Invoice(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    file = models.FileField(upload_to='invoices/')
    vendor = models.CharField(max_length=255)
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    date = models.DateField()
    subtotal = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    tax_total = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    status = models.CharField(max_length=50, choices=[
        ('Pending', 'Pending'),
        ('Exported', 'Exported'),
        ('Failed', 'Failed')
    ], default='Pending')

    created_at = models.DateTimeField(auto_now_add=True)

    objects = InvoiceManager()

    def __str__(self):
        return f"{self.vendor} - {self.amount} on {self.date}"

    def get_full_details(self) -> dict:
        """
        Returns a dictionary with all invoice details, parsing the file if necessary.
        This encapsulates the logic of falling back to the parser if DB fields are empty.
        """
        from .services.parser import parse_invoice_details # Local import to avoid circular dependency
        import os
        from django.conf import settings

        # Start with persisted data from the database
        line_items = [
            {
                'description': li.description, 'quantity': li.quantity,
                'unit_price': li.unit_price, 'line_total': li.line_total,
            }
            for li in self.lines.all().order_by('order', 'id')
        ]
        subtotal = self.subtotal
        tax_total = self.tax_total

        # If data is incomplete, parse the file as a fallback
        if subtotal is None or tax_total is None or not line_items:
            file_path = self.file.path if hasattr(self.file, 'path') else os.path.join(settings.MEDIA_ROOT, self.file.name)
            if os.path.exists(file_path):
                parsed_details = parse_invoice_details(file_path)
                subtotal = subtotal if subtotal is not None else parsed_details.get('subtotal')
                tax_total = tax_total if tax_total is not None else parsed_details.get('tax_total')
                if not line_items:
                    line_items = parsed_details.get('line_items') or []
        
        return {
            'subtotal': subtotal, 'tax_total': tax_total, 'line_items': line_items,
        }


class InvoiceLine(models.Model):
    invoice = models.ForeignKey(Invoice, on_delete=models.CASCADE, related_name='lines')
    description = models.CharField(max_length=512, blank=True)
    quantity = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    unit_price = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    line_total = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    order = models.IntegerField(null=True, blank=True)

    def __str__(self):
        return f"Line for {self.invoice_id}: {self.description[:30] if self.description else ''}"
