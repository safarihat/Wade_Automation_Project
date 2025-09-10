from django.db import models
from django.contrib.auth.models import User

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

    def __str__(self):
        return f"{self.vendor} - {self.amount} on {self.date}"

class InvoiceLine(models.Model):
    invoice = models.ForeignKey(Invoice, on_delete=models.CASCADE, related_name='lines')
    description = models.CharField(max_length=512, blank=True)
    quantity = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    unit_price = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    line_total = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    order = models.IntegerField(null=True, blank=True)

    def __str__(self):
        return f"Line for {self.invoice_id}: {self.description[:30] if self.description else ''}"

