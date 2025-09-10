from django import forms

class ContactForm(forms.Form):
    name = forms.CharField(
        max_length=100,
        required=True,
        widget=forms.TextInput(attrs={'placeholder': 'e.g. John Doe'})
    )
    company_name = forms.CharField(
        max_length=100,
        required=False,
        widget=forms.TextInput(attrs={'placeholder': 'e.g. Acme Ltd'})
    )
    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={'placeholder': 'e.g. john.doe@example.com'})
    )
    issue = forms.CharField(
        required=True,
        widget=forms.Textarea(attrs={
            'rows': 6,
            'placeholder': 'Please tell us about the administrative task you want to automate...'
        })
    )

