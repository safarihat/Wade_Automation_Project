from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm

from django.contrib.auth import logout as auth_logout
from .models import ClientProfile
from django.core.mail import send_mail
from django.contrib import messages
from django.conf import settings
from .forms import ContactForm

# ... any other views you have ...

def contact(request):
    if request.method == 'POST':
        form = ContactForm(request.POST)
        if form.is_valid():
            name = form.cleaned_data['name']
            company = form.cleaned_data.get('company_name', 'N/A')
            from_email = form.cleaned_data['email']
            issue = form.cleaned_data['issue']

            subject = f'New Contact Form Submission from {name}'
            
            # Note: The 'from_email' is the user's email, but for many email providers,
            # you must send from an authorized address. We'll use the user's email
            # in the message body and the reply-to header.
            message_body = f"""
            You have a new message from your website contact form:

            Name: {name}
            Company: {company}
            Email: {from_email}

            Message:
            {issue}
            """

            try:
                send_mail(
                    subject=subject,
                    message=message_body,
                    from_email=settings.DEFAULT_FROM_EMAIL,
                    recipient_list=[settings.ADMIN_EMAIL], # Your notification email
                    fail_silently=False,
                    # Set the user's email as the reply-to address
                    reply_to=[from_email] 
                )
                messages.success(request, 'Thank you for your message! We will get back to you shortly.')
                return redirect('home')
            except Exception:
                messages.error(request, 'Sorry, there was an error sending your message. Please try again later.')

    else:
        form = ContactForm()

    return render(request, 'wade_automation/contact.html', {'form': form})


def home(request):
    form = AuthenticationForm()
    return render(request, 'wade_automation/home.html', {'form': form})

def about(request):
    return render(request, 'wade_automation/about.html')

def pathways_view(request):
    return render(request, 'wade_automation/pathways.html')

@login_required
def customer_dashboard(request):
    if request.user.is_superuser:
        from django.urls import reverse
        return redirect(reverse('admin:index'))  # Superusers go to Django admin

    # Ensure the user has a ClientProfile; create one automatically if missing
    profile, _ = ClientProfile.objects.get_or_create(
        user=request.user,
        defaults={'company_name': 'New Customer'}
    )

    # Build a list of installed modules for this user. Keep minimal coupling by importing lazily.
    installed_modules = []

    from django.urls import reverse
    # Add Document Generator module to the dashboard
    installed_modules.append({
        'key': 'doc_generator',
        'name': 'Document Generator',
        'description': 'Create regulatory documents using AI-powered templates.',
        'url': reverse('doc_generator:plan_wizard_start'), # Prefer the new wizard entrypoint
        'icon': None,
    })

    try:
        from invoice_reconciliation.views import has_access as invoice_has_access
    except Exception:
        invoice_has_access = None

    # Add Invoice Reconciliation tile if accessible
    if invoice_has_access and invoice_has_access(request.user):
        installed_modules.append({
            'key': 'invoice_reconciliation',
            'name': 'Invoice Reconciliation',
            'description': 'Upload, view, export, and push invoices to Xero.',
            'url': '/invoices/dashboard/',
            'icon': None,  # You can add a static icon path later
        })

    context = {
        'user': request.user,
        'profile': profile,
        'installed_modules': installed_modules,
    }
    return render(request, 'wade_automation/dashboard.html', context)

@login_required
def logout_view(request):
    auth_logout(request)
    return redirect('home')
