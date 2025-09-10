from django import forms
from .models import FreshwaterPlan

class FreshwaterPlanForm(forms.ModelForm):
    class Meta:
        model = FreshwaterPlan
        fields = ['latitude', 'longitude', 'council'] # Added council as it's a required field in the model

    def clean_latitude(self):
        latitude = self.cleaned_data['latitude']
        if not (-90 <= latitude <= 90):
            raise forms.ValidationError("Latitude must be between -90 and 90 degrees.")
        return latitude

    def clean_longitude(self):
        longitude = self.cleaned_data['longitude']
        if not (-180 <= longitude <= 180):
            raise forms.ValidationError("Longitude must be between -180 and 180 degrees.")
        return longitude
