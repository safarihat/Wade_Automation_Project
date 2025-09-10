from django import forms

class InvoiceUploadForm(forms.Form):
    file = forms.FileField(
        label='Upload Invoice',
        help_text='Accepted formats: PDF, PNG, JPG',
        widget=forms.ClearableFileInput(attrs={'class': 'form-control'})
    )
