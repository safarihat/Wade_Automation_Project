# invoice_reconciliation/services/xero_exporter.py

def push_to_xero(invoices):
    # Placeholder logic for now
    # You can integrate with Xero API later
    for invoice in invoices:
        invoice.status = 'Exported'
        invoice.save()
    return True
