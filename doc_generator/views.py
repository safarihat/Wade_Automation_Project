from django.views.generic.edit import CreateView
from django.views.generic.detail import DetailView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.urls import reverse_lazy
from django.contrib import messages
from django.contrib.gis.geos import Point
from django.conf import settings
from .models import FreshwaterPlan, RegionalCouncil
from .forms import FreshwaterPlanForm
# Assuming generate_plan_task is defined in doc_generator/tasks.py
from .tasks import generate_plan_task # This import assumes tasks.py exists

import logging

logger = logging.getLogger(__name__)

class FreshwaterPlanCreateView(LoginRequiredMixin, CreateView):
    model = FreshwaterPlan
    form_class = FreshwaterPlanForm
    template_name = 'doc_generator/freshwater_plan_form.html' # Assuming this template path

    def get_success_url(self):
        # Redirect to the preview page of the object that was just created
        return reverse_lazy('doc_generator:freshwater_plan_preview', kwargs={'pk': self.object.pk})

    def form_valid(self, form):
        logger.info("Form is valid, calling celery task.")
        # Get latitude and longitude from the cleaned form data
        lat = form.cleaned_data.get('latitude')
        lon = form.cleaned_data.get('longitude')
        
        # Create a GEOS Point object for the spatial query
        location_point = Point(lon, lat, srid=4326) if lat is not None and lon is not None else None

        # Perform the spatial query to find the containing council.
        try:
            council = RegionalCouncil.objects.get(geom__contains=location_point)
        except RegionalCouncil.DoesNotExist:
            messages.error(self.request, "Could not find a regional council for the provided coordinates. Please check the location.")
            return self.form_invalid(form)
        except RegionalCouncil.MultipleObjectsReturned:
            # This can happen with overlapping polygons; we'll just take the first one.
            council = RegionalCouncil.objects.filter(geom__contains=location_point).first()

        # Set the user, council, and location on the form instance before saving.
        # The default ModelForm.save() will now handle saving all fields correctly.
        form.instance.user = self.request.user
        form.instance.council = council.name
        form.instance.location = location_point

        # Now, call the parent's form_valid, which will call form.save()
        response = super().form_valid(form) # This sets self.object

        # Call the Celery task to generate the plan in the background
        generate_plan_task.delay(self.object.pk)
        return response

    def form_invalid(self, form):
        logger.error(f"Form is invalid. Errors: {form.errors.as_json()}")
        return super().form_invalid(form)

class FreshwaterPlanPreviewView(LoginRequiredMixin, DetailView):
    model = FreshwaterPlan
    template_name = 'doc_generator/freshwater_plan_preview.html'
    context_object_name = 'plan'

    def get_queryset(self):
        # Ensure users can only see their own plans
        return FreshwaterPlan.objects.filter(user=self.request.user)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['LINZ_API_KEY'] = settings.LINZ_API_KEY
        return context