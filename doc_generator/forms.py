from django import forms
from .models import FreshwaterPlan

class FreshwaterPlanForm(forms.ModelForm):
    coordinates = forms.CharField(
        label="Latitude and Longitude",
        help_text="Enter comma-separated latitude and longitude (e.g., -41.2865, 174.7762)"
    )

    class Meta:
        model = FreshwaterPlan
        fields = ['council']

    def clean_coordinates(self):
        coordinates = self.cleaned_data['coordinates']
        try:
            lat_str, lon_str = coordinates.split(',')
            latitude = float(lat_str.strip())
            longitude = float(lon_str.strip())
        except ValueError:
            raise forms.ValidationError("Invalid format. Please use comma-separated latitude and longitude.")

        if not (-90 <= latitude <= 90):
            raise forms.ValidationError("Latitude must be between -90 and 90 degrees.")
        if not (-180 <= longitude <= 180):
            raise forms.ValidationError("Longitude must be between -180 and 180 degrees.")

        self.cleaned_data['latitude'] = latitude
        self.cleaned_data['longitude'] = longitude
        return coordinates

    def save(self, commit=True):
        instance = super().save(commit=False)
        instance.latitude = self.cleaned_data['latitude']
        instance.longitude = self.cleaned_data['longitude']
        if commit:
            instance.save()
        return instance
