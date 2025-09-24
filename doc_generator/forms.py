import re
from django import forms
from django.contrib.gis.geos import Point
from doc_generator.models import FreshwaterPlan
from crispy_forms.helper import FormHelper
from django.urls import reverse, reverse_lazy
from crispy_forms.layout import Layout, Fieldset, Row, Column, Div, HTML
from crispy_forms.bootstrap import PrependedText, FormActions

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

    def __init__(self, *args, **kwargs):
        """
        Override the init method to make latitude and longitude not required
        at the form level. This allows our custom clean() method to run and
        populate them from the 'geolocation_paste' field before final validation.
        """
        super().__init__(*args, **kwargs)
        self.fields['latitude'].required = False
        self.fields['longitude'].required = False

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


class LocationForm(forms.Form):
    """
    Lightweight form used in the Step 1 wizard page to accept a pasted
    coordinate string (latitude, longitude). The page script parses this
    and fills hidden fields used on submit.
    """
    geolocation_paste = forms.CharField(
        label="Paste Google Maps Location",
        required=True,
        widget=forms.TextInput(attrs={
            'placeholder': 'e.g., -41.2865, 174.7762'
        })
    )


class AdminDetailsForm(forms.ModelForm):
    """
    Form for Step 2 of the wizard, allowing users to review and edit
    the administrative details of their farm plan.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_method = 'post'
        self.helper.form_id = 'admin-details-form'
        self.helper.form_tag = True
        self.helper.form_class = 'form-vertical'
        self.helper.label_class = 'form-label fw-bold'
        self.helper.field_class = 'mb-3'

        self.helper.layout = Layout(
            Fieldset(
                'People & Business',
                'operator_name',
                'owner_name',
                'operator_nzbn',
                'plan_preparer_name',
                'operator_contact_details',
                'owner_contact_details',
            ),
            Fieldset(
                'Property & Location Details',
                'farm_address',
                'council_authority_name',
                'legal_land_titles',
                'total_farm_area_ha',
                'leased_area_ha',
                'land_use',
                'resource_consents',
            ),
            FormActions(
                HTML('<a href="{}" class="btn btn-lg btn-outline-secondary"><i class="fas fa-chevron-left me-2"></i>Back</a>'.format(reverse_lazy('doc_generator:plan_wizard_start'))),
                HTML('<button type="submit" class="btn btn-lg btn-gradient ms-3">Save & Continue<i class="fas fa-chevron-right ms-2"></i></button>'),
                css_class="d-flex justify-content-end mt-5"
            )
        )

        # Make the AI-populated field read-only
        self.fields['council_authority_name'].widget.attrs['readonly'] = True
        self.fields['farm_address'].widget.attrs['readonly'] = True


    class Meta:
        model = FreshwaterPlan
        fields = [
            'operator_name',
            'operator_contact_details',
            'operator_nzbn',
            'owner_name',
            'owner_contact_details',
            'plan_preparer_name',
            'farm_address',
            'council_authority_name',
            'legal_land_titles',
            'total_farm_area_ha',
            'leased_area_ha',
            'land_use',
            'resource_consents',
        ]
