import re
from django import forms
from django.contrib.gis.geos import Point
from .models import FreshwaterPlan

class FreshwaterPlanForm(forms.ModelForm):
    """
    A form for creating or updating a Freshwater Plan.
    The user provides a single location string pasted from Google Maps.
    """
    geolocation_paste = forms.CharField(
        label="Paste Google Maps Location",
        required=True,
        widget=forms.TextInput(attrs={
            'placeholder': 'e.g., -41.2865, 174.7762'
        })
    )

    class Meta:
        model = FreshwaterPlan
        # The form needs to know about the model fields it will populate,
        # even if they are hidden from the user. This is crucial for validation.
        fields = ['latitude', 'longitude']
        widgets = {
            'latitude': forms.HiddenInput(),
            'longitude': forms.HiddenInput(),
        }

    def clean(self):
        cleaned_data = super().clean()
        geoloc_string = cleaned_data.get('geolocation_paste')

        if not geoloc_string:
            return cleaned_data

        # Use a more robust regex to handle integers and floats
        numbers = re.findall(r'-?\d+(?:\.\d+)?', geoloc_string)

        if len(numbers) == 2:
            lat, lon = float(numbers[0]), float(numbers[1])
            
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                self.add_error('geolocation_paste', "Invalid latitude or longitude values.")
            else:
                # This populates the data for the hidden fields
                cleaned_data['latitude'] = lat
                cleaned_data['longitude'] = lon
        else:
            self.add_error('geolocation_paste', "Could not parse coordinates. Please use the format 'latitude, longitude'.")

        return cleaned_data
