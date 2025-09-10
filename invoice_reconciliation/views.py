from django.shortcuts import render

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
from decimal import Decimal
from datetime import date

from .forms import InvoiceUploadForm
from .models import Invoice, InvoiceLine
from .services.parser import parse_invoice_file, parse_invoice_details
from .services.xero_exporter import push_to_xero

# Check if user has access to this module
# For now: allow superusers and any user with a ClientProfile
# Replace with a stricter check (e.g., profile flag) when ready

def has_access(user):
    if user.is_superuser:
        return True
    return hasattr(user, 'clientprofile')

@login_required
def upload_invoice(request):
    if not has_access(request.user):
        return redirect('dashboard')

    if request.method == 'POST':
        form = InvoiceUploadForm(request.POST, request.FILES)
        if form.is_valid():
            files = request.FILES.getlist('file')
            if not files and 'file' in request.FILES:
                files = [request.FILES['file']]

            pending = []
            for file in files:
                saved_rel_path = default_storage.save(f'invoices/{file.name}', file)
                abs_path = os.path.join(settings.MEDIA_ROOT, saved_rel_path)
                details = parse_invoice_details(abs_path)
                # Prepare line items text block for easy editing
                line_items = details.get('line_items') or []
                lines_text = "\n".join([
                    f"{(li.get('description') or '').replace('|','/')}|{li.get('quantity') or ''}|{li.get('unit_price') or ''}|{li.get('line_total') or ''}"
                    for li in line_items
                ])
                pending.append({
                    'file_path': saved_rel_path,
                    'filename': os.path.basename(saved_rel_path),
                    'vendor': details.get('vendor') or 'Unknown Vendor',
                    'amount': str(details.get('amount')) if details.get('amount') is not None else '0.00',
                    'date': details.get('date').isoformat() if details.get('date') else '',
                    'subtotal': str(details.get('subtotal')) if details.get('subtotal') is not None else '',
                    'tax_total': str(details.get('tax_total')) if details.get('tax_total') is not None else '',
                    'lines_text': lines_text,
                })
            request.session['pending_invoices'] = pending
            return redirect('review_invoices')
    else:
        form = InvoiceUploadForm()

    return render(request, 'invoice_reconciliation/upload.html', {'form': form})

@login_required
def invoice_dashboard(request):
    if not has_access(request.user):
        return redirect('dashboard')

    invoices = Invoice.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'invoice_reconciliation/dashboard.html', {'invoices': invoices})

@login_required
def export_invoices_csv(request):
    if not has_access(request.user):
        return redirect('dashboard')

    invoices = Invoice.objects.filter(user=request.user)
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="invoices.csv"'

    writer = csv.writer(response)
    writer.writerow([
        'Vendor', 'Invoice Date', 'Subtotal', 'Tax Total', 'Grand Total', 'Status', 'Line Items (Description|Qty|Unit Price|Line Total; ...)'
    ])
    for inv in invoices:
        # prefer persisted fields and persisted lines
        subtotal = inv.subtotal
        tax_total = inv.tax_total
        line_items_qs = getattr(inv, 'lines', None)
        line_items = []
        if line_items_qs is not None:
            for li in inv.lines.all().order_by('order', 'id'):
                line_items.append({
                    'description': li.description,
                    'quantity': li.quantity,
                    'unit_price': li.unit_price,
                    'line_total': li.line_total,
                })
        if subtotal is None or tax_total is None or not line_items:
            # fallback parse
            details = parse_invoice_details(inv.file.path if hasattr(inv.file, 'path') else os.path.join(settings.MEDIA_ROOT, inv.file.name))
            subtotal = subtotal if subtotal is not None else details.get('subtotal')
            tax_total = tax_total if tax_total is not None else details.get('tax_total')
            if not line_items:
                line_items = details.get('line_items') or []
        items_str = "; ".join([
            f"{(str(it.get('description') or '')).replace(';',' ').strip()}|{it.get('quantity') or ''}|{it.get('unit_price') or ''}|{it.get('line_total') or ''}"
            for it in line_items
        ])
        writer.writerow([
            inv.vendor,
            inv.date,
            subtotal or '',
            tax_total or '',
            inv.amount,
            inv.status,
            items_str
        ])

    return response

@login_required
def push_invoices_to_xero(request):
    if not has_access(request.user):
        return redirect('dashboard')
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
def clear_invoices(request):
    if not has_access(request.user):
        return redirect('dashboard')
    if request.method == 'POST':
        Invoice.objects.filter(user=request.user).delete()
        messages.success(request, 'All your invoices were deleted.')
    else:
        messages.error(request, 'Invalid request method for clear action.')
    return redirect('invoice_dashboard')

@login_required
def review_invoices(request):
    if not has_access(request.user):
        return redirect('dashboard')
    pending = request.session.get('pending_invoices', [])
    if not pending:
        messages.info(request, 'No uploaded invoices to review.')
        return redirect('invoice_dashboard')
    
    for p in pending:
        p['file_url'] = os.path.join(settings.MEDIA_URL, p['file_path']).replace('\\', '/')

    return render(request, 'invoice_reconciliation/review.html', {'pending': pending})

@login_required
def confirm_invoices(request):
    if not has_access(request.user):
        return redirect('dashboard')
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

    created_count = 0
    for i in range(len(files)):
        rel_path = files[i]
        vendor = vendors[i].strip() or 'Unknown Vendor'
        amt_str = (amounts[i] or '0.00').strip()
        sub_str = (subtotals[i] or '').strip()
        tax_str = (taxes[i] or '').strip()
        dt_str = (dates[i] or '').strip()

        try:
            amt = Decimal(amt_str)
        except Exception:
            amt = Decimal('0.00')
        sub = None
        tax = None
        try:
            if sub_str:
                sub = Decimal(sub_str)
        except Exception:
            sub = None
        try:
            if tax_str:
                tax = Decimal(tax_str)
        except Exception:
            tax = None
        inv_date = parse_date(dt_str) or parse_date(str(dt_str))
        if inv_date is None:
            inv_date = date.today()

        inv = Invoice.objects.create(
            user=request.user,
            file=rel_path,
            vendor=vendor,
            amount=amt,
            date=inv_date,
            subtotal=sub,
            tax_total=tax,
            status='Pending'
        )
        # Parse line items textarea
        text_block = lines_texts[i] or ''
        order = 1
        for row in text_block.splitlines():
            row = row.strip()
            if not row:
                continue
            parts = row.split('|')
            # Ensure 4 parts
            while len(parts) < 4:
                parts.append('')
            desc = parts[0].strip()
            q = parts[1].strip()
            up = parts[2].strip()
            lt = parts[3].strip()
            def to_dec(s):
                try:
                    return Decimal(s) if s else None
                except Exception:
                    return None
            InvoiceLine.objects.create(
                invoice=inv,
                description=desc[:512],
                quantity=to_dec(q),
                unit_price=to_dec(up),
                line_total=to_dec(lt),
                order=order
            )
            order += 1
        created_count += 1

    request.session['pending_invoices'] = []
    messages.success(request, f'{created_count} invoice(s) saved.')
    return redirect('invoice_dashboard')

@login_required
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

