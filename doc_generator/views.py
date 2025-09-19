import json
import requests
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.urls import reverse
from django.contrib import messages
from django.contrib.gis.geos import Point
from django import forms
from django.conf import settings
from django.http import JsonResponse, HttpResponseBadRequest
from doc_generator.forms import AdminDetailsForm, LocationForm
from doc_generator.models import FreshwaterPlan, RegionalCouncil
from doc_generator.tasks import generate_plan_task, populate_admin_details_task

import logging

logger = logging.getLogger(__name__)


@login_required
def freshwater_plan_preview(request, pk):
    """
    Displays a preview of the generated freshwater plan.
    Handles POST requests for saving drawn features for planned works.
    """
    plan = get_object_or_404(FreshwaterPlan, pk=pk, user=request.user)

    if request.method == 'POST':
        # This part handles saving the 'planned_works_features' from the preview page map.
        planned_works_data = request.POST.get('planned_works_features')
        if planned_works_data:
            try:
                plan.activity_features = json.loads(planned_works_data)
                plan.save(update_fields=['activity_features', 'updated_at'])
                messages.success(request, "Your planned works have been saved.")
            except json.JSONDecodeError:
                messages.error(request, "There was an error saving your planned works. Invalid data format.")
        else:
            plan.activity_features = None
            plan.save(update_fields=['activity_features', 'updated_at'])
            messages.info(request, "Your planned works have been cleared.")
        return redirect('doc_generator:freshwater_plan_preview', pk=plan.pk)

    return render(request, 'doc_generator/freshwater_plan_preview.html', {
        'plan': plan,

        'LINZ_API_KEY': settings.LINZ_API_KEY,
        'drawn_features_json': json.dumps(plan.vulnerability_features) if plan.vulnerability_features else 'null',
        'planned_works_features_json': json.dumps(plan.activity_features) if plan.activity_features else 'null',
    })


# =============================================================================
# NEW MULTI-STEP WIZARD VIEWS
# =============================================================================

@login_required
def plan_wizard_start(request):
    """
    Step 1 of the wizard: Confirm the farm's location.
    """
    if request.method == 'POST':
        # This POST comes from the confirmation button, with coords in hidden fields
        lat = request.POST.get('latitude')
        lon = request.POST.get('longitude')

        if lat and lon:
            location_point = Point(float(lon), float(lat), srid=4326)
            # Synchronously find the council to satisfy the NOT NULL constraint if DB is out of sync
            council_obj = RegionalCouncil.objects.filter(geom__contains=location_point).first()

            # Create the initial plan object
            plan = FreshwaterPlan.objects.create(
                user=request.user,
                latitude=float(lat),
                longitude=float(lon),
                location=location_point,
                council=council_obj.name if council_obj else "Unknown Council",
                farm_address="Address pending...", # Placeholder for out-of-sync DB
                land_use="Pending..." # Placeholder for out-of-sync DB
            )
            # Trigger the FAST background task to populate admin details
            populate_admin_details_task.delay(plan.pk)
            return redirect(reverse('doc_generator:plan_wizard_details', kwargs={'pk': plan.pk}))

        else:
            # Handle case where coordinates are missing
            form = LocationForm(request.POST)
            form.add_error(None, "Could not find coordinates to proceed.")
    else:
        form = LocationForm()

    context = {
        'form': form,

        'LINZ_API_KEY': settings.LINZ_API_KEY,
        'LINZ_BASEMAPS_API_KEY': settings.LINZ_BASEMAPS_API_KEY,
    }
    return render(request, 'doc_generator/plan_step1_location.html', context)


@login_required
def plan_wizard_details(request, pk):
    """
    Step 2 of the wizard. Shows a loading page while AI runs, then shows
    the pre-populated form for user review.
    """
    plan = get_object_or_404(FreshwaterPlan, pk=pk, user=request.user)

    if request.method == 'POST':
        # This is for when the user SUBMITS their corrections
        form = AdminDetailsForm(request.POST, instance=plan)
        if form.is_valid():
            plan.save()
            # Now trigger the SLOW task to generate the full plan in the background
            generate_plan_task.delay(plan.pk)
            messages.success(request, "Your details have been saved.")
            return redirect(reverse('doc_generator:plan_wizard_map_vulnerabilities', kwargs={'pk': plan.pk}))
        else:
            # If form is invalid, re-render the details page with errors
            context = {'form': form, 'plan': plan}
            return render(request, 'doc_generator/plan_step2_details.html', context)
    
    # This is for the GET request.
    # Check for a query parameter to see if we are in the final review step.
    if request.GET.get('step') == 'review':
        # We've been redirected here after the plan was ready. Show the form.
        form = AdminDetailsForm(instance=plan)
        return render(request, 'doc_generator/plan_step2_details.html', {'form': form, 'plan': plan})

    # If no query param, check the status.
    if plan.generation_status == FreshwaterPlan.GenerationStatus.READY:
        # The plan is ready. Redirect to this same view, but add the query
        # parameter. This ensures the transaction is committed and visible.
        redirect_url = f"{reverse('doc_generator:plan_wizard_details', kwargs={'pk': pk})}?step=review"
        return redirect(redirect_url)
    elif plan.generation_status == FreshwaterPlan.GenerationStatus.FAILED:
        messages.error(request, "There was an error generating plan details. Please try again.")
        return redirect('doc_generator:plan_wizard_start')
    else:
        # AI is still working, show the loading page
        context = {'plan': plan}
        return render(request, 'doc_generator/plan_step2_loading.html', context)


@login_required
def check_plan_status(request, pk):
    """
    An API endpoint for the loading page to poll.
    Returns the generation status of the plan.
    """
    plan = get_object_or_404(FreshwaterPlan, pk=pk, user=request.user)
    return JsonResponse({'status': plan.generation_status})


@login_required
def get_parcel_geometry(request):
    """
    An API endpoint that takes latitude and longitude, queries the LINZ WFS
    for the intersecting property parcel, and returns its geometry as GeoJSON.
    """
    lat = request.GET.get('lat')
    lon = request.GET.get('lon')

    if not lat or not lon:
        return HttpResponseBadRequest("Missing 'lat' or 'lon' parameters.")

    wfs_url = f"https://data.linz.govt.nz/services;key={settings.LINZ_API_KEY}/wfs"
    params = {
        'service': 'WFS',
        'version': '2.0.0',
        'request': 'GetFeature',
        'typeNames': 'layer-50772',  # NZ Primary Parcels
        'outputFormat': 'application/json',
        'srsName': 'urn:ogc:def:crs:EPSG::4326', # Ensure output is in WGS84
        'cql_filter': f"INTERSECTS(shape, SRID=4326;POINT({lon} {lat}))"
    }

    try:
        response = requests.get(wfs_url, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()

        if data and data.get('features'):
            # Return the entire GeoJSON FeatureCollection
            return JsonResponse(data)
        else:
            # Return an empty FeatureCollection if no parcel is found
            return JsonResponse({'type': 'FeatureCollection', 'features': []})

    except requests.RequestException as e:
        logger.error(f"WFS request for parcel geometry failed: {e}")
        return JsonResponse({'error': 'Failed to retrieve parcel data from LINZ.'}, status=502)


@login_required
def plan_wizard_map_vulnerabilities(request, pk):
    """
    Step 3 of the wizard: Map inherent vulnerabilities. (Placeholder)
    """
    plan = get_object_or_404(FreshwaterPlan, pk=pk, user=request.user)

    if request.method == 'POST':
        vulnerability_data = request.POST.get('vulnerability_features')
        if vulnerability_data:
            try:
                # Validate that it's proper JSON before saving
                plan.vulnerability_features = json.loads(vulnerability_data)
                messages.success(request, "Vulnerability map features have been saved.")
            except json.JSONDecodeError:
                messages.error(request, "There was an error saving your map data. Invalid format.")
        else:
            # If the user clears all features, save it as null
            plan.vulnerability_features = None
            messages.info(request, "Vulnerability map features have been cleared.")
        
        plan.save(update_fields=['vulnerability_features', 'updated_at'])
        return redirect(reverse('doc_generator:plan_wizard_map_activities', kwargs={'pk': plan.pk}))

    context = {
        'plan': plan,

        'LINZ_API_KEY': settings.LINZ_API_KEY,
        'vulnerability_features_json': json.dumps(plan.vulnerability_features) if plan.vulnerability_features else 'null',
    }
    return render(request, 'doc_generator/plan_step3_map_vulnerabilities.html', context)


@login_required
def plan_wizard_map_activities(request, pk):
    """
    Step 4 of the wizard: Map farming activities. (Placeholder)
    """
    plan = get_object_or_404(FreshwaterPlan, pk=pk, user=request.user)
    # TODO: Implement the form and logic for the activities map.
    context = {'plan': plan}
    return render(request, 'doc_generator/plan_step4_map_activities.html', context)
