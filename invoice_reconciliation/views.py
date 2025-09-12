from django.shortcuts import render

import logging
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, JsonResponse, FileResponse
from django.core.files.storage import default_storage
from django.conf import settings
from django.urls import reverse
from django.contrib import messages
from django.utils.dateparse import parse_date
import os
import csv
from decimal import Decimal, InvalidOperation
from datetime import date
from typing import Optional

from .forms import InvoiceUploadForm
from .models import Invoice, InvoiceLine
from .services.parser import parse_invoice_file, parse_invoice_details
from .decorators import user_has_access
from .services.xero_exporter import push_to_xero

def _to_decimal(s: str) -> Optional[Decimal]:
    """Safely convert a string to a Decimal, returning None on failure."""
    try:
        # Ensure the input is a string and not empty before converting
        return Decimal(str(s)) if s else None
    except (InvalidOperation, TypeError, ValueError):
        return None

def _prepare_pending_invoice_data(file) -> dict:
    """Saves an uploaded file, parses it, and returns a dictionary for session storage."""
    saved_rel_path = default_storage.save(f'invoices/{file.name}', file)
    abs_path = os.path.join(settings.MEDIA_ROOT, saved_rel_path)
    details = parse_invoice_details(abs_path)

    # Prepare line items text block for easy editing in the review form
    line_items = details.get('line_items') or []
    lines_text = "\n".join([
        f"{(li.get('description') or '').replace('|','/')}|{li.get('quantity') or ''}|{li.get('unit_price') or ''}|{li.get('line_total') or ''}"
        for li in line_items
    ])

    return {
        'file_path': saved_rel_path,
        'filename': os.path.basename(saved_rel_path),
        'vendor': details.get('vendor') or 'Unknown Vendor',
        'amount': str(details.get('amount')) if details.get('amount') is not None else '0.00',
        'date': details.get('date').isoformat() if details.get('date') else '',
        'subtotal': str(details.get('subtotal')) if details.get('subtotal') is not None else '',
        'tax_total': str(details.get('tax_total')) if details.get('tax_total') is not None else '',
        'lines_text': lines_text,
    }

@login_required
@user_has_access
def upload_invoice(request):
    if request.method == 'POST':
        form = InvoiceUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # request.FILES.getlist() correctly handles both single and multiple file uploads.
            files = request.FILES.getlist('file')
            request.session['pending_invoices'] = [_prepare_pending_invoice_data(f) for f in files]
            return redirect('review_invoices')
    else:
        form = InvoiceUploadForm()

    return render(request, 'invoice_reconciliation/upload.html', {'form': form})

@login_required
@user_has_access
def invoice_dashboard(request):
    invoices = Invoice.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'invoice_reconciliation/dashboard.html', {'invoices': invoices})

@login_required
@user_has_access
def export_invoices_csv(request):
    invoices = Invoice.objects.filter(user=request.user)
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="invoices.csv"'

    writer = csv.writer(response)
    writer.writerow([
        'Vendor', 'Invoice Date', 'Subtotal', 'Tax Total', 'Grand Total', 'Status', 'Line Items (Description|Qty|Unit Price|Line Total; ...)'
    ])
    for inv in invoices:
        # Use the new model method to get all details, with fallback logic included.
        details = inv.get_full_details()
        
        # Format the line items into a single string for the CSV.
        items_str = "; ".join([
            f"{(str(it.get('description') or '')).replace(';',' ').strip()}|{it.get('quantity') or ''}|{it.get('unit_price') or ''}|{it.get('line_total') or ''}"
            for it in details['line_items']
        ])

        writer.writerow([
            inv.vendor,
            inv.date,
            details['subtotal'] or '',
            details['tax_total'] or '',
            inv.amount,
            inv.status,
            items_str
        ])

    return response

@login_required
@user_has_access
def push_invoices_to_xero(request):
    if request.method != 'POST':
        messages.error(request, 'Invalid request method for push action.')
        return redirect('invoice_dashboard')

    invoices = Invoice.objects.filter(user=request.user, status='Pending')
    result = push_to_xero(invoices)
    if result:
        messages.success(request, f'Successfully pushed {invoices.count()} invoices to Xero (placeholder).')
    else:
        messages.error(request, 'Failed to push invoices to Xero.')
    return redirect('invoice_dashboard')

@login_required
@user_has_access
def clear_invoices(request):
    if request.method == 'POST':
        Invoice.objects.filter(user=request.user).delete()
        messages.success(request, 'All your invoices were deleted.')
    else:
        messages.error(request, 'Invalid request method for clear action.')
    return redirect('invoice_dashboard')

@login_required
@user_has_access
def review_invoices(request):
    pending = request.session.get('pending_invoices', [])
    if not pending:
        messages.info(request, 'No uploaded invoices to review.')
        return redirect('invoice_dashboard')
    
    for p in pending:
        # Use string formatting for URL construction to avoid OS-specific path separators.
        p['file_url'] = f"{settings.MEDIA_URL}{p['file_path']}"

    return render(request, 'invoice_reconciliation/review.html', {'pending': pending})

@login_required
@user_has_access
def confirm_invoices(request):
    if request.method != 'POST':
        messages.error(request, 'Invalid request method.')
        return redirect('invoice_dashboard')

    # Expect fields arrays aligned by index
    files = request.POST.getlist('file_path')
    vendors = request.POST.getlist('vendor')
    amounts = request.POST.getlist('amount')
    dates = request.POST.getlist('date')
    subtotals = request.POST.getlist('subtotal')
    taxes = request.POST.getlist('tax_total')
    lines_texts = request.POST.getlist('lines_text')

    # Add a sanity check to ensure all form data lists are of the same length.
    num_invoices = len(files)
    if not all(len(lst) == num_invoices for lst in [vendors, amounts, dates, subtotals, taxes, lines_texts]):
        messages.error(request, "There was a mismatch in the submitted invoice data. Please review and try again.")
        # Redirect back to the review page, as the session data is still there.
        return redirect('review_invoices')

    created_count = 0
    for i in range(len(files)):
        rel_path = files[i]
        vendor = vendors[i].strip() or 'Unknown Vendor'
        amt_str = (amounts[i] or '0.00').strip()
        sub_str = (subtotals[i] or '').strip()
        tax_str = (taxes[i] or '').strip()
        dt_str = (dates[i] or '').strip()

        # Use the helper for consistent and safe decimal conversion
        amt = _to_decimal(amt_str) or Decimal('0.00') # Fallback to 0.00 for the main amount
        sub = _to_decimal(sub_str)
        tax = _to_decimal(tax_str)
        inv_date = parse_date(dt_str) or parse_date(str(dt_str))
        if inv_date is None:
            inv_date = date.today()

        invoice_data = {
            'file': rel_path,
            'vendor': vendor,
            'amount': amt,
            'date': inv_date,
            'subtotal': sub,
            'tax_total': tax,
            'status': 'Pending'
        }
        lines_text = lines_texts[i] or ''

        try:
            Invoice.objects.create_with_lines(request.user, invoice_data, lines_text)
            created_count += 1
        except Exception as e:
            # Catch any error during the creation of a single invoice,
            # log it, and inform the user, so the other invoices can still be processed.
            logging.error(f"Failed to save invoice for vendor '{vendor}': {e}", exc_info=True)
            messages.error(request, f"Could not save invoice for '{vendor}'. An unexpected error occurred.")
            continue # Skip to the next invoice

    request.session['pending_invoices'] = []
    messages.success(request, f'{created_count} invoice(s) saved.')
    return redirect('invoice_dashboard')

@login_required
@user_has_access
def download_invoice(request, invoice_id: int):
    try:
        inv = Invoice.objects.get(id=invoice_id, user=request.user)
    except Invoice.DoesNotExist:
        messages.error(request, 'Invoice not found.')
        return redirect('invoice_dashboard')
    file_path = inv.file.path if hasattr(inv.file, 'path') else os.path.join(settings.MEDIA_ROOT, inv.file.name)
    if not os.path.exists(file_path):
        messages.error(request, 'File not found on server.')
        return redirect('invoice_dashboard')
    return FileResponse(open(file_path, 'rb'), as_attachment=True, filename=os.path.basename(file_path))

@login_required
@user_has_access
def delete_invoice(request, invoice_id: int):
    if request.method != 'POST':
        messages.error(request, 'Invalid request method for delete.')
        return redirect('invoice_dashboard')
    try:
        inv = Invoice.objects.get(id=invoice_id, user=request.user)
    except Invoice.DoesNotExist:
        messages.error(request, 'Invoice not found.')
        return redirect('invoice_dashboard')
    inv.delete()
    messages.success(request, 'Invoice deleted.')
    return redirect('invoice_dashboard')
