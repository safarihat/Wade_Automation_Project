import json
import time
import requests
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.urls import reverse
from django.contrib import messages
from django.contrib.gis.geos import Point
from django.db import transaction
from django.views import View
import pyproj
from django.conf import settings
from django.http import JsonResponse, HttpResponseBadRequest
from doc_generator.forms import AdminDetailsForm, LocationForm
from doc_generator.utils import (
    _query_koordinates_vector,
    _query_arcgis_vector,
    _query_koordinates_raster,
    _calculate_slope_from_dem,
)
from doc_generator.models import FreshwaterPlan, RegionalCouncil
from doc_generator.tasks import generate_plan_task, populate_admin_details_task

# LangChain components for RAG - needed for the new analysis view
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from groq import AuthenticationError
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser

import logging

logger = logging.getLogger(__name__)

# The user requested this to be added.
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
@transaction.atomic
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
            form.save()
            # Now trigger the SLOW task to generate the full plan in the background
            # generate_plan_task.delay(plan.pk) # This can be enabled when ready
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
    return JsonResponse({
        'status': plan.generation_status,
        'progress': plan.generation_progress or []
    })


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



class NZLRIErosionLayerView(View):
    """
    An API endpoint that proxies requests to the LRIS API for the 
    NZLRI Erosion Type and Severity layer (48054), handling API key authentication.
    """
    # Data Source: Landcare Research, licensed under the Landcare Data Use License.
    
    def get(self, request, *args, **kwargs):
        lat = request.GET.get('lat')
        lon = request.GET.get('lon')

        if not lat or not lon:
            return HttpResponseBadRequest("Missing 'lat' or 'lon' parameters.")

        if not settings.LRIS_API_KEY:
            logger.error("LRIS_API_KEY is not configured in environment settings.")
            return JsonResponse({'error': 'API key for LRIS service is not configured.'}, status=500)

        query_url = "https://lris.scinfo.org.nz/services/query/v1/vector.json/"
        params = {
            'v': '1.2',
            'layer': 48054,
            'x': lon,
            'y': lat,
            'radius': 10000,
            'max_results': 100,
            'geometry': 'true',
            'with_field_names': 'true',
            'key': settings.LRIS_API_KEY
        }

        try:
            response = requests.get(query_url, params=params, timeout=30)
            response.raise_for_status()
            geojson_data = response.json()
            
            # GeoJSON standard (RFC 7946) specifies coordinates MUST be in WGS84 (EPSG:4326).
            # We assume the service complies and returns WGS84 GeoJSON, so no transformation is needed.
            return JsonResponse(geojson_data)

        except json.JSONDecodeError:
            logger.error(f"LRIS API returned non-JSON response. Status: {response.status_code}, Content: {response.text[:500]}...") # Log first 500 chars
            return JsonResponse(
                {'error': 'Failed to parse LRIS API response as JSON. It might be returning an HTML error page.'},
                status=500
            )
        except requests.exceptions.HTTPError as e:
            logger.error(f"LRIS API request failed with status {e.response.status_code}: {e.response.text}")
            return JsonResponse(
                {'error': f'Failed to fetch data from LRIS service. Status: {e.response.status_code}'}, 
                status=e.response.status_code
            )
        except requests.RequestException as e:
            logger.error(f"LRIS API request failed: {e}")
            return JsonResponse({'error': f'Failed to fetch data from LRIS service: {str(e)}'}, status=502)


class ProtectedAreasLayerView(View):
    """
    API endpoint to proxy requests for DOC Public Conservation Areas from Koordinates.
    """
    def get(self, request, *args, **kwargs):
        lat = request.GET.get('lat')
        lon = request.GET.get('lon')
        if not lat or not lon:
            return HttpResponseBadRequest("Missing 'lat' or 'lon' parameters.")

        try:
            koordinates_api_key = settings.KOORDINATES_API_KEY
            if not koordinates_api_key:
                logger.error("Koordinates API key is not configured.")
                return JsonResponse({'error': 'Koordinates API key is not configured.'}, status=500)

            layer_id = 754  # DOC Public Conservation Areas
            features = _query_koordinates_vector(layer_id, float(lon), float(lat), koordinates_api_key, radius=10000, max_results=50)

            if features is None:
                return JsonResponse({'error': 'Failed to retrieve data from Koordinates service.', 'details': 'The query function returned None.'}, status=502)

            # The helper already returns a list of feature-like dicts. Wrap in a FeatureCollection.
            feature_collection = {'type': 'FeatureCollection', 'features': features}
            return JsonResponse(feature_collection)
        except Exception as e:
            logger.exception("Error fetching data from Koordinates for ProtectedAreasLayerView")
            return JsonResponse({"error": "Koordinates proxy failed", "details": str(e)}, status=500)


class GroundwaterZonesLayerView(View):
    """
    API endpoint to proxy requests for Groundwater Management Zones from Environment Southland.
    """
    def get(self, request, *args, **kwargs):
        lat = request.GET.get('lat')
        lon = request.GET.get('lon')
        if not lat or not lon:
            return HttpResponseBadRequest("Missing 'lat' or 'lon' parameters.")
        
        try:
            # The _query_arcgis_vector helper function is perfect for this.
            # It handles point-in-polygon queries and returns GeoJSON-like features.
            arcgis_url = 'https://maps.es.govt.nz/server/rest/services/Public/WaterAndLand/MapServer/3/query'
            features = _query_arcgis_vector(arcgis_url, float(lon), float(lat))

            # The helper returns a list of features. We wrap it in a FeatureCollection.
            # The helper also ensures geometry is included, which is needed for the map.
            feature_collection = {'type': 'FeatureCollection', 'features': features or []}
            return JsonResponse(feature_collection)
        except Exception as e:
            logger.exception("Error fetching data from ArcGIS for GroundwaterZonesLayerView")
            return JsonResponse({"error": "ArcGIS proxy failed for Groundwater Zones", "details": str(e)}, status=500)


@login_required
def plan_wizard_map_vulnerabilities(request, pk):
    """
    Step 3 of the wizard: Map inherent vulnerabilities.
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
        'LINZ_BASEMAPS_API_KEY': settings.LINZ_BASEMAPS_API_KEY,
        'KOORDINATES_API_KEY': settings.KOORDINATES_API_KEY,
        'LRIS_API_KEY': settings.LRIS_API_KEY,
        'vulnerability_features_json': json.dumps(plan.vulnerability_features) if plan.vulnerability_features else 'null',
    }
    return render(request, 'doc_generator/plan_step3_map_vulnerabilities.html', context)


@login_required
def plan_wizard_map_activities(request, pk):
    """
    Step 4 of the wizard: Map farming activities. (Placeholder)
    Allows users to draw features like paddocks, tracks, and infrastructure.
    """
    plan = get_object_or_404(FreshwaterPlan, pk=pk, user=request.user)

    if request.method == 'POST':
        activity_data = request.POST.get('activity_features')
        if activity_data:
            try:
                plan.activity_features = json.loads(activity_data)
                messages.success(request, "Farming activity map features have been saved.")
            except json.JSONDecodeError:
                messages.error(request, "There was an error saving your map data. Invalid format.")
        else:
            plan.activity_features = None
            messages.info(request, "Farming activity map features have been cleared.")
        
        plan.save(update_fields=['activity_features', 'updated_at'])
        # Redirect to the next step when it's ready. For now, we can redirect to a placeholder or preview.
        return redirect(reverse('freshwater_plan_preview', kwargs={'pk': plan.pk})) # Placeholder redirect

    context = {
        'plan': plan,
        'LINZ_BASEMAPS_API_KEY': settings.LINZ_BASEMAPS_API_KEY,
        'activity_features_json': json.dumps(plan.activity_features) if plan.activity_features else 'null',
    }
    return render(request, 'doc_generator/plan_step4_map_activities.html', context)


# =============================================================================
# KOORDINATES API DATA FETCHING (Replaces WFS)
# =============================================================================

@login_required
def api_generate_vulnerability_analysis(request, pk):
    """
    API endpoint to generate a freshwater vulnerability analysis using RAG and
    the Koordinates REST API for geospatial data.
    """
    plan = get_object_or_404(FreshwaterPlan, pk=pk, user=request.user)
    
    # Use the dedicated Koordinates API key for all Koordinates.com queries.
    koordinates_api_key = settings.KOORDINATES_API_KEY
    if not koordinates_api_key:
        logger.error("Koordinates API key is not configured.")
        return JsonResponse({'error': 'Koordinates API key is not configured.'}, status=500)

    # Add a check for the Groq API key to prevent crashes and provide a clear error.
    if not settings.GROQ_API_KEY:
        logger.error("Groq AI API key (GROQ_API_KEY) is not configured in your .env file.")
        return JsonResponse({'error': 'Groq AI API key (GROQ_API_KEY) is not configured in your .env file.'}, status=500)

    # Define the layers to be queried. We now mix Koordinates and ArcGIS sources.
    # Using example URLs for Southland. These should be verified.
    # Koordinates Layer IDs: NZ 8m DEM (51768), NZ 8m Slope (51769)
    DATA_LAYERS = {
        'regional_soil': {'type': 'arcgis_vector', 'name': 'Regional Soil Data', 'url': 'https://services3.arcgis.com/v5RzLI7nHYeFImL4/arcgis/rest/services/Freshwater_farm_plan_contextual_data_hosted/FeatureServer/5/query'},
        # --- Koordinates Vector Layers ---
        'land_cover': {'id': 117733, 'type': 'koordinates_raster', 'name': 'LCDB v5.0 Land Cover'},
        'erosion': {'id': 48054, 'type': 'vector', 'name': 'NZLRI Erosion Type and Severity', 'radius': 1000},
        'protected_areas': {'id': 754, 'type': 'vector', 'name': 'DOC Public Conservation Areas', 'radius': 1000},
        'land_parcels': {'id': 53682, 'type': 'vector', 'name': 'NZ Land Parcels', 'radius': 100},
        'groundwater_zones': {'type': 'arcgis_vector', 'name': 'Groundwater Management Zones', 'url': 'https://maps.es.govt.nz/server/rest/services/Public/WaterAndLand/MapServer/3/query'},
    }

    try:
        # 1. Fetch data from all configured Koordinates layers
        geospatial_context = {}
        for key, layer_info in DATA_LAYERS.items():
            layer_name = layer_info['name']
            logger.info(f"Querying data layer: {layer_name}")
            
            result_obj = {"layer_name": layer_name}

            result = None # Initialize result to None
            if layer_info['type'] == 'arcgis_vector':
                result = _query_arcgis_vector(layer_info['url'], plan.longitude, plan.latitude)
            elif layer_info['type'] == 'vector': # Koordinates vector
                layer_id = layer_info['id']
                result_obj["layer_id"] = layer_id
                radius = layer_info.get('radius', 100)
                result = _query_koordinates_vector(layer_id, plan.longitude, plan.latitude, koordinates_api_key, radius=radius)
            elif layer_info['type'] == 'koordinates_raster':
                result = _query_koordinates_raster(layer_info['id'], plan.longitude, plan.latitude, koordinates_api_key)
            
            # Safely update result_obj, handling None from _query functions
            if result is not None:
                if isinstance(result, list) and layer_info['type'] in ['arcgis_vector', 'vector']:
                    result_obj["features"] = result
                elif isinstance(result, dict) and layer_info['type'] == 'koordinates_raster':
                    # The function now returns a dict like {'value': 123}.
                    # We'll keep this structure for the LLM.
                    result_obj["data"] = result
                else:
                    logger.warning(f"Unexpected result type from {layer_name}: {type(result)}")
                    result_obj["features" if "vector" in layer_info['type'] else "values"] = {}
            else:
                logger.info(f"No data or error for layer {layer_name}, returning empty fallback.")
                if layer_info['type'] in ['arcgis_vector', 'vector']:
                    result_obj["features"] = []
                elif layer_info['type'] == 'koordinates_raster':
                    result_obj["data"] = {}

            geospatial_context[key] = result_obj
            time.sleep(1) # Rate limiting for the free tier

        # Always calculate slope from DEM
        logger.info("Calculating slope from DEM for analysis.")
        slope_degrees = _calculate_slope_from_dem(plan.longitude, plan.latitude, koordinates_api_key)
        geospatial_context['calculated_slope'] = {'slope_degrees': slope_degrees}


        # 2. Format the fetched data for the LLM prompt
        def format_context(data_obj):
            """Serializes the structured data object into a JSON string for the LLM."""
            # We no longer need to create a human-readable summary here.
            # Instead, we pass the structured JSON directly to the LLM.
            # The LLM will be instructed to handle "info" or "error" keys.
            # We remove the geometry to save tokens, as the LLM can't use it directly.
            if 'features' in data_obj:
                for feature in data_obj['features']:
                    if 'geometry' in feature:
                        del feature['geometry']
            return json.dumps(data_obj, indent=2)

        soil_context = format_context(geospatial_context.get('regional_soil', {}))
        land_cover_context = format_context(geospatial_context.get('land_cover', {}))
        erosion_context = format_context(geospatial_context.get('erosion', {}))
        protection_context = format_context(geospatial_context.get('protected_areas', {}))
        slope_context = format_context(geospatial_context.get('calculated_slope', {}))
        parcels_context = format_context(geospatial_context.get('land_parcels', {}))
        groundwater_context = format_context(geospatial_context.get('groundwater_zones', {}))

        # 3. Set up and use the RAG chain to get regional policy context
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': 'cpu'})
        vector_store_path = settings.BASE_DIR / 'vector_store'
        vector_store = Chroma(persist_directory=str(vector_store_path), embedding_function=embeddings)
        retriever = vector_store.as_retriever(search_kwargs={'k': 3})
        llm = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=settings.GROQ_API_KEY)
        output_parser = JsonOutputParser()
        
        template = """You are an expert environmental consultant specializing in New Zealand's freshwater regulations.
        Your task is to analyze site-specific data and produce a JSON object containing a technical summary and a structured table of biophysical vulnerabilities.
        The primary context for regulations and environmental values comes from the retrieved documents for the specific catchment: **{catchment_name}**.
        Synthesize the following information. Each data source is provided as a JSON object.
        1.  **Data Source 1: Regional Policy Documents for {council}**
        {context}
        
        2.  **Data Source 2: Site-Specific Soil Data (from regional council)** (JSON object)
        {soil_data}

        3.  **Data Source 3: Site-Specific Slope Data (calculated from DEM)** (JSON object)
        {slope_data}

        4.  **Data Source 4: Site-Specific Land Parcel Data (from LINZ)** (JSON object)
        {parcels_data}

        5.  **Data Source 5: Site-Specific Land Cover Data (LCDB)** (JSON object)
        {land_cover_data}

        6.  **Data Source 6: Site-Specific Erosion Risk Data (National HEL)** (JSON object)
        {erosion_data}

        7.  **Data Source 7: Site-Specific Protected Area Status (National DOC)** (JSON object)
        {protection_data}

        8.  **Data Source 8: Site-Specific Groundwater Zone Data (from Environment Southland)** (JSON object)
        {groundwater_data}

        ---
        **Task:**
        Based on all the provided data, generate a JSON object that follows this exact format:
        {{
            "technical_summary": "A high-level, markdown-formatted overview summarizing the most critical findings from all data sources (1-8). This should be a brief executive summary of the property's environmental context and key risks. Refer to the maps and their legends for visual context.",
            "catchment_context_summary": "A markdown-formatted text summary of the key catchment-level context, based ONLY on the retrieved regional policy documents (Source 1). Explain the {catchment_name}'s identity, its main environmental challenges, and key values (e.g., cultural, ecological, recreational). Explicitly draw correlations between the catchment context and the technical data presented in the 'technical_summary' (e.g., 'The high erosion risk identified in the technical summary is particularly concerning given the catchment's sensitivity to sediment runoff as highlighted in regional policies.').",
            "layer_explanations": {{
                "land_use": "A markdown explanation of what the Land Cover (LCDB) data (Source 5) shows for the property and its immediate surroundings. Describe the dominant land cover types and their potential implications, referring to the visual representation on the map.",
                "erosion": "A markdown explanation of the erosion risk based on the NZLRI data (Source 6). Describe the severity and type of erosion identified on or near the property.",
                "protected_areas": "A markdown explanation of any nearby DOC Public Conservation Areas (Source 7). State whether the property intersects or is near any protected areas and what that might imply.",
                "groundwater_zones": "A markdown explanation of the groundwater management zone(s) the property is in, based on the data from Environment Southland (Source 8). Describe the zone name, its lithology, and any implications."
            }},
            "biophysical_table": [
                {{
                    "feature": "Climate",
                    "considerations": ["A list of considerations based on the data, e.g., 'High annual rainfall noted in regional context.'"],
                    "vulnerabilities": ["A list of resulting vulnerabilities, e.g., 'Increased risk of flooding.', 'Potential for sheet erosion.']
                }},
                {{
                    "feature": "Landform",
                    "considerations": ["e.g., 'DEM-derived slope is 12 degrees (Source: Koordinates DEM).', 'ArcGIS data indicates SlopeAngle is 11 (Source: Regional Soil Data).'],
                    "vulnerabilities": ["e.g., 'Moderate risk of mass movement erosion on steeper faces.']
                }}
            ]
        }}
        
        **Instructions for Interpretation:**
        - For 'technical_summary': Provide a brief, high-level overview synthesizing all available information.
        - For 'layer_explanations': For each key ('land_use', 'erosion', etc.), provide a detailed, markdown-formatted paragraph explaining what the corresponding data source reveals about the property itself. Be specific and contextual.
        - Populate the `biophysical_table` with entries for Climate, Landform, Soil, and other relevant features based on the provided data.
        - If data for a feature is missing (e.g., no soil data found), reflect this in the considerations and vulnerabilities (e.g., "No site-specific soil data available.").
        - The final output MUST be a single, valid JSON object and nothing else.
        """
        prompt = PromptTemplate.from_template(template)

        # This is the correct LCEL structure for a RAG chain.
        # The retriever is invoked with the catchment name to get specific context.
        rag_chain = (
            {
                "context": lambda x: retriever.invoke(x["catchment_name"]),
                "catchment_name": lambda x: x["catchment_name"],
                "council": lambda x: plan.council,
                "soil_data": lambda x: soil_context,
                "land_cover_data": lambda x: land_cover_context,
                "erosion_data": lambda x: erosion_context,
                "protection_data": lambda x: protection_context,
                "slope_data": lambda x: slope_context,
                "parcels_data": lambda x: parcels_context,
                "groundwater_data": lambda x: groundwater_context
            }
            | prompt
            | llm
            | output_parser
        )

        analysis_data = rag_chain.invoke({"catchment_name": plan.catchment_name})

        return JsonResponse(analysis_data)

    except requests.exceptions.ConnectionError as e:
        if 'Failed to resolve' in str(e):
            logger.error(f"DNS resolution failed for vulnerability analysis: {e}", exc_info=True)
            return JsonResponse({'error': 'Network Error: Could not resolve the address of an external mapping service. Please check your internet connection and DNS settings.'}, status=504)
        logger.error(f"Connection error during vulnerability analysis: {e}", exc_info=True)
        return JsonResponse({'error': 'A network connection error occurred while contacting an external mapping service.'}, status=504)
    except AuthenticationError as e:
        logger.error(f"Groq AI authentication failed: {e}", exc_info=True)
        return JsonResponse({'error': 'Groq AI authentication failed. Please check your GROQ_API_KEY in the .env file.'}, status=500)
    except ImportError as e:
        # This can happen if the groq library is not installed or has issues.
        if 'groq' in str(e).lower():
            logger.error(f"Groq library import error: {e}", exc_info=True)
            return JsonResponse({'error': 'The Groq AI library is not installed correctly. Please check your environment.'}, status=500)
        raise # Re-raise other import errors
    except Exception as e:
        logger.error(f"Error in vulnerability analysis for plan {pk}: {e}", exc_info=True)
        return JsonResponse({'error': 'An unexpected error occurred while generating the analysis.'}, status=500)