from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_invoice, name='upload_invoice'),
    path('review/', views.review_invoices, name='review_invoices'),
    path('confirm/', views.confirm_invoices, name='confirm_invoices'),
    path('dashboard/', views.invoice_dashboard, name='invoice_dashboard'),
    path('export/', views.export_invoices_csv, name='export_invoices_csv'),
    path('push/', views.push_invoices_to_xero, name='push_invoices_to_xero'),
    path('clear/', views.clear_invoices, name='clear_invoices'),
    path('download/<int:invoice_id>/', views.download_invoice, name='download_invoice'),
    path('delete/<int:invoice_id>/', views.delete_invoice, name='delete_invoice'),
]
