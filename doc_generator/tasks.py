import os
import io
import json
import math
import logging
import time
import requests
import django_rq
from django.conf import settings
from django.db import transaction
from django.template.loader import render_to_string
from django.core.files.base import ContentFile
from doc_generator.models import FreshwaterPlan
from .geospatial_utils import (
    transform_coords,
    _query_koordinates_vector,
    _query_arcgis_vector,
    _query_koordinates_raster,
    _query_arcgis_raster,
    _calculate_slope_from_dem,
)
from doc_generator.services.soil_drainage_service import SoilDrainageService
from .services.vulnerability_service import VulnerabilityService
import hashlib
from langchain_core.documents import Document
import redis
from django_rq import job
from rq import get_current_job
from rq.job import Job, Retry



# PDF and Watermarking libraries
PDF_DEPS_AVAILABLE = True
try:
    from xhtml2pdf import pisa
    from pypdf import PdfReader, PdfWriter
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
except ImportError as e:
    PDF_DEPS_AVAILABLE = False
    pisa = None
    PdfReader = None
    PdfWriter = None
    canvas = None
    inch = None
    colors = None
    letter = None

# LangChain components for RAG
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from .services.embedding_service import get_embedding_model, _embedding_model
from .models import MonitoringSite

# Setup logger
logger = logging.getLogger(__name__)

def _get_vulnerability_service(plan_pk: int, site_context_dict: dict = None) -> VulnerabilityService:
    """
    Centralized factory function to instantiate the VulnerabilityService.
    Ensures the service is created consistently with all dependencies in each task.
    """
    from .models import FreshwaterPlan
    from langchain_chroma import Chroma
    from .services.embedding_service import get_embedding_model
    import os
    from django.conf import settings

    # Initialize retriever
    embeddings = get_embedding_model()
    vector_store_path = os.path.join(settings.BASE_DIR, 'vector_store')
    vector_store = Chroma(persist_directory=vector_store_path, embedding_function=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})

    # Get site context
    if site_context_dict:
        site_context = site_context_dict
    else:
        plan = FreshwaterPlan.objects.get(pk=plan_pk)
        site_context = {"pk": plan.pk, "plan_pk": plan.pk, "council_authority_name": plan.council_authority_name, "catchment_name": plan.catchment_name}

    return VulnerabilityService(retriever=retriever, site_context=site_context)


def _create_watermark_pdf() -> io.BytesIO:
    """Creates a PDF in memory containing only the 'PREVIEW ONLY' watermark."""
    packet = io.BytesIO()
    can = canvas.Canvas(packet, pagesize=letter)
    can.saveState()
    can.setFont('Helvetica-Bold', 60)
    can.setFillColor(colors.grey, alpha=0.2)
    can.translate(letter[0] / 2, letter[1] / 2)
    can.rotate(45)
    can.drawCentredString(0, 0, "PREVIEW ONLY")
    can.restoreState()
    can.save()
    packet.seek(0)
    return packet


def _stamp_pdf(pdf_content: bytes, watermark_pdf: io.BytesIO) -> io.BytesIO:
    """Stamps a watermark onto an existing PDF's content."""
    output_pdf_stream = io.BytesIO()
    writer = PdfWriter()

    main_pdf = PdfReader(io.BytesIO(pdf_content))
    watermark_page = PdfReader(watermark_pdf).pages[0]

    for page in main_pdf.pages:
        page.merge_page(watermark_page)
        writer.add_page(page)

    writer.write(output_pdf_stream)
    output_pdf_stream.seek(0)
    return output_pdf_stream


@job('default')
def generate_plan_task(freshwater_plan_id):
    """
    RQ job to generate a freshwater plan using a RAG pipeline,
    then create a watermarked PDF preview.
    """
    logger.info(f"--- Starting plan generation for FreshwaterPlan ID: {freshwater_plan_id} ---")
    try:
        freshwater_plan = FreshwaterPlan.objects.get(pk=freshwater_plan_id)

        _generate_plan_text(freshwater_plan)
        _generate_static_map_image(freshwater_plan)
        _generate_pdf_preview(freshwater_plan)

        logger.info(f"--- Plan generation and PDF preview complete for ID: {freshwater_plan_id} ---")

    except FreshwaterPlan.DoesNotExist:
        logger.error(f"Task failed: FreshwaterPlan with ID {freshwater_plan_id} does not exist.")
    except Exception as e:
        logger.error(f"An error occurred during plan generation for ID {freshwater_plan_id}: {e}", exc_info=True)
        # RQ will handle retry based on queue settings, or you can re-enqueue manually
        raise e


def _generate_plan_text(freshwater_plan: FreshwaterPlan):
    """Generates the text content of the plan and saves it to the model."""
    if freshwater_plan.generated_plan:
        logger.info(f"Plan text for ID {freshwater_plan.pk} already exists. Skipping generation.")
        return

    logger.info(f"[1/3] Initializing RAG components for plan ID: {freshwater_plan.pk}")
    embeddings = get_embedding_model()
    vector_store_path = os.path.join(settings.BASE_DIR, 'vector_store')
    vector_store = Chroma(persist_directory=vector_store_path, embedding_function=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={'k': 5})
    llm = OllamaLLM(model="phi3:mini", base_url="http://localhost:11434", options={"num_ctx": 2048})

    template = """
    You are an expert assistant for creating New Zealand Freshwater Farm Plans.
    Your task is to generate a section of a freshwater farm plan based on the provided context and user information.
    The plan is for a location within the '{council}' regional council area.

    Use the following retrieved context to answer the question. The context contains relevant regulations, policies, and guidelines.
    Focus on creating a practical, well-structured, and compliant report.
    If the context is insufficient, state what information is missing rather than inventing details.

    Context:
    {context}

    ---
    Question:
    {question}

    Generated Plan Section:
    """
    prompt = PromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough(), "council": lambda x: freshwater_plan.council}
        | prompt
        | llm
        | StrOutputParser()
    )
    logger.info(f"RAG components initialized for plan ID: {freshwater_plan.pk}")

    logger.info(f"[2/3] Invoking RAG chain for plan ID: {freshwater_plan.pk}")
    question_for_llm = f"""
    Generate a preliminary freshwater farm plan for a property with the following details:
    - Address: {freshwater_plan.farm_address}
    - Legal Land Titles: {freshwater_plan.legal_land_titles}
    - Regional Council: {freshwater_plan.council}

    Focus on identifying key environmental risks and suggesting initial mitigation strategies based on the provided context.
    """
    generated_content = rag_chain.invoke(question_for_llm)
    logger.info(f"RAG chain invocation complete for plan ID: {freshwater_plan.pk}")

    logger.info(f"[3/3] Saving generated text to database for plan ID: {freshwater_plan.pk}")
    freshwater_plan.generated_plan = generated_content
    freshwater_plan.save(update_fields=['generated_plan', 'updated_at'])
    logger.info(f"Plan text generation complete for ID: {freshwater_plan.pk}")


def _generate_static_map_image(freshwater_plan: FreshwaterPlan):
    """
    Fetches a static map image from the LINZ Data Service (LDS) WMS
    and saves it to the FreshwaterPlan's map_image field.
    """
    if freshwater_plan.map_image:
        logger.info(f"Map image for ID {freshwater_plan.pk} already exists. Skipping generation.")
        return
    logger.info(f"Generating static map image for plan ID: {freshwater_plan.pk}")

    try:
        lon_nztm, lat_nztm = transform_coords(freshwater_plan.longitude, freshwater_plan.latitude, 4326, 2193)
    except Exception as e:
        logger.error(f"Coordinate transformation failed for plan {freshwater_plan.pk}: {e}")
        return

    half_size = 500
    bbox_nztm = (
        lon_nztm - half_size,
        lat_nztm - half_size,
        lon_nztm + half_size,
        lat_nztm + half_size,
    )

    wms_url = "https://basemaps.linz.govt.nz/v1/wms"
    params = {
        'api': settings.LINZ_BASEMAPS_API_KEY,
        'service': 'WMS',
        'request': 'GetMap',
        'layers': 'aerial',
        'styles': '',
        'format': 'image/png',
        'transparent': 'true',
        'version': '1.1.1',
        'width': 800,
        'height': 800,
        'srs': 'EPSG:2193',
        'bbox': ','.join(map(str, bbox_nztm)),
    }

    try:
        response = requests.get(wms_url, params=params, timeout=30)
        response.raise_for_status()

        file_name = f"map_plan_{freshwater_plan.pk}.png"
        freshwater_plan.map_image.save(file_name, ContentFile(response.content), save=False)
        freshwater_plan.save(update_fields=['map_image', 'updated_at'])
        logger.info(f"Saved static map image for plan ID: {freshwater_plan.pk}")
    except requests.exceptions.RequestException as e:
        logger.warning(f"Could not fetch static map for plan ID {freshwater_plan.pk}: {e}")


def _generate_pdf_preview(freshwater_plan: FreshwaterPlan):
    """
    Renders the plan to HTML, converts to PDF, and stamps a watermark.
    """
    if freshwater_plan.pdf_preview:
        logger.info(f"PDF preview for ID {freshwater_plan.pk} already exists. Skipping generation.")
        return

    if not PDF_DEPS_AVAILABLE:
        logger.warning("PDF generation libraries not installed; skipping PDF preview generation.")
        return

    logger.info(f"Generating PDF preview for plan ID: {freshwater_plan.pk}")

    html_string = render_to_string('doc_generator/pdf_template.html', {'plan': freshwater_plan})

    pdf_stream = io.BytesIO()
    pisa_status = pisa.CreatePDF(io.StringIO(html_string), dest=pdf_stream)
    if pisa_status.err:
        raise Exception(f"PDF generation failed with xhtml2pdf: {pisa_status.err}")
    pdf_bytes = pdf_stream.getvalue()

    watermark = _create_watermark_pdf()
    final_pdf_stream = _stamp_pdf(pdf_bytes, watermark)

    file_name = f"preview_plan_{freshwater_plan.pk}.pdf"
    file_content = final_pdf_stream.read()
    freshwater_plan.pdf_preview.save(file_name, ContentFile(file_content), save=False)
    freshwater_plan.save(update_fields=['pdf_preview', 'updated_at'])

    logger.info(f"Saved watermarked PDF preview for plan ID: {freshwater_plan.pk}")


def _update_progress(plan_id: int, message: str, status: str = "pending"):
    """
    Helper function to append a progress update to the plan's log.
    """
    try:
        plan = FreshwaterPlan.objects.get(pk=plan_id)
        if not isinstance(plan.generation_progress, list):
            plan.generation_progress = []

        plan.generation_progress.append({"message": message, "status": status})
        plan.save(update_fields=['generation_progress', 'updated_at'])
    except FreshwaterPlan.DoesNotExist:
        logger.warning(f"_update_progress called for non-existent plan ID {plan_id}")


# --- Sub-tasks for Parallel Execution ---
@job('high')
def get_council_authority_name_task(council_name):
    """
    Verifies and retrieves the official council authority name using a RAG query.
    Falls back to the input name if the RAG service is unavailable.
    """
    job = get_current_job()
    job_id = job.id if job else 'ad-hoc'
    logger.info(f"Task get_council_authority_name_task started for council: {council_name}", extra={'job_id': job_id})

    if not council_name:
        logger.warning("get_council_authority_name_task received an empty council name.", extra={'job_id': job_id})
        return "Unknown Council"

    try:
        # The RAGService abstracts away the details of Chroma, embeddings, and the LLM.
        from .services.rag_service import RAGService
        rag_service = RAGService()
        verified_name = rag_service.query_council_authority(council_name)
        logger.info(f"Successfully verified council '{council_name}' as '{verified_name}'.", extra={'job_id': job_id})
        return verified_name or council_name
    except ConnectionRefusedError as e:
        logger.error(f"Could not connect to RAG service (ChromaDB) to verify council '{council_name}'. Is the service running? Error: {e}", extra={'job_id': job_id})
        return council_name # Graceful fallback
    except Exception as e:
        logger.error(f"An unexpected error occurred during council name verification for '{council_name}': {e}", exc_info=True, extra={'job_id': job_id})
        return council_name # Graceful fallback


@job('high')
def get_address_task(lat, lon):
    headers = {'User-Agent': 'WadeAutomation/1.0 (contact@example.com)'}
    url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}"
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    return response.json().get('display_name', "Address not found.")


@job('high')
def get_catchment_task(lon, lat):
    fmu_url = 'https://services3.arcgis.com/v5RzLI7nHYeFImL4/arcgis/rest/services/Freshwater_farm_plan_contextual_data_hosted/FeatureServer/1/query'
    data = _query_arcgis_vector(fmu_url, lon, lat)
    if isinstance(data, list) and data:
        raw_name = data[0].get('properties', {}).get('Zone', 'Not found')
        return raw_name.lower().replace('catchment', '').strip().title()
    return "Catchment/FMU not found."


def _get_parcel_features_from_wfs(lat, lon):
    """
    Attempts to fetch parcel features from a WFS endpoint as a fallback.
    Returns a list of features or None.
    """
    try:
        wfs_url = "https://data.linz.govt.nz/services;key={api_key}/wfs".format(api_key=settings.KOORDINATES_API_KEY)
        params = {
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetFeature", # NZ Primary Parcels layer ID
            "typeNames": "layer-50772", # Koordinates WFS uses "layer-<id>" for typeNames
            "outputFormat": "application/json",
            "srsName": "EPSG:4326",
            "cql_filter": f"INTERSECTS(geometry,POINT({lon} {lat}))"
        }
        response = requests.get(wfs_url, params=params, timeout=15)
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as http_err:
            logger.warning(f"Failed to fetch parcel features from WFS: {http_err}. Response content: {response.text}")
            return None
        data = response.json()
        return data.get("features", [])
    except Exception as e:
        logger.warning(f"Failed to fetch parcel features from WFS: {e}")
        return None


@job('high')
def get_parcel_task(lon, lat):
    parcel_layer_id = 53682
    parcel_data = _query_koordinates_vector(parcel_layer_id, lon, lat, settings.KOORDINATES_API_KEY, radius=50)
    if isinstance(parcel_data, list) and parcel_data:
        titles = ", ".join([f.get('properties', {}).get('appellation', '') for f in parcel_data])
        area = sum(f.get('properties', {}).get('area_ha', 0) for f in parcel_data if f.get('properties', {}).get('area_ha'))
        return {"titles": titles, "area": area}

    wfs_parcel_data = _get_parcel_features_from_wfs(lat, lon)
    if wfs_parcel_data:
        titles = ", ".join([f.get('properties', {}).get('appellation', '') for f in wfs_parcel_data])
        area = sum(f.get('properties', {}).get('survey_area_ha', 0) for f in wfs_parcel_data if f.get('properties', {}).get('survey_area_ha'))
        return {"titles": titles, "area": area}

    return {"titles": "No parcel information found.", "area": None}


@job('high')
def get_soil_data_task(lon, lat):
    soil_url = 'https://services3.arcgis.com/v5RzLI7nHYeFImL4/arcgis/rest/services/Freshwater_farm_plan_contextual_data_hosted/FeatureServer/5/query'
    soil_data = _query_arcgis_vector(soil_url, lon, lat)
    if isinstance(soil_data, list) and soil_data:
        attributes = soil_data[0].get('properties', {})
        return {
            "soil_type": attributes.get('SoilType', 'Unknown Soil'),
            "arcgis_slope_angle": attributes.get('SlopeAngle'),
            "nutrient_leaching_vulnerability": attributes.get('NutrientLeachingVulnerability'),
            "erodibility": attributes.get('Erodibility')
        }
    return {}


@job('high')
def get_soil_drainage_task(lat, lon):
    drainage_service = SoilDrainageService(lat=lat, lon=lon)
    return drainage_service.get_soil_drainage_class()


@job('high')
def get_slope_class_task(lon, lat):
    slope_degrees = _calculate_slope_from_dem(lon, lat, settings.KOORDINATES_API_KEY)
    if slope_degrees is not None:
        if slope_degrees < 3:
            slope_class = 'Flat (0-3°)'
        elif slope_degrees < 7:
            slope_class = 'Gently Undulating (3-7°)'
        elif slope_degrees < 15:
            slope_class = 'Rolling (7-15°)'
        else:
            slope_class = 'Steep (>15°)'
        return f"{slope_class} ({slope_degrees:.1f}°)", round(slope_degrees, 1)
    return "Not available.", None


@job('default')
def process_fetched_data_task(plan_id):
    """
    Processes the results from the parallel data fetching tasks
    and saves them to the FreshwaterPlan model.
    """
    current_job = get_current_job()
    redis_conn = current_job.connection
    
    try:
        plan = FreshwaterPlan.objects.get(pk=plan_id)
        
        # Fetch results from parent jobs
        dependency_ids = current_job.dependency_ids
        dependency_jobs = Job.fetch_many(dependency_ids, connection=redis_conn)
        
        # Use a dictionary to store results, keyed by the function name.
        # This is more robust than relying on the order of the jobs.
        results = {job.func_name.split('.')[-1]: job.result for job in dependency_jobs if job.result is not None}

        # Safely get results from the dictionary with fallbacks.
        council_authority_name_result = results.get('get_council_authority_name_task', "Could not verify council name.") # New result
        address_result = results.get('get_address_task', "Address not found.")
        catchment_result = results.get('get_catchment_task', "Catchment/FMU not found.")
        parcel_result = results.get('get_parcel_task', {"titles": "No parcel information found.", "area": None})
        soil_data_result = results.get('get_soil_data_task', {})
        soil_drainage_result = results.get('get_soil_drainage_task', "Not available.")
        
        # The slope task returns a tuple, so handle it carefully.
        slope_result_tuple = results.get('get_slope_class_task', ("Not available.", None))
        if isinstance(slope_result_tuple, tuple) and len(slope_result_tuple) == 2:
            slope_class_result, slope_angle_dem = slope_result_tuple
        else:
            # Fallback in case the result format is unexpected
            slope_class_result, slope_angle_dem = "Not available.", None

        plan.council_authority_name = council_authority_name_result # New line
        _update_progress(plan.pk, f"Council authority name retrieved: {council_authority_name_result}", "complete") # Updated progress message

        plan.farm_address = address_result
        _update_progress(plan.pk, "Address retrieved.", "complete")

        plan.catchment_name = catchment_result
        
        # Ensure parcel_result is a dictionary before accessing keys
        if not isinstance(parcel_result, dict):
            parcel_result = {"titles": "Parcel data in unexpected format.", "area": None}
            
        plan.legal_land_titles = parcel_result["titles"]
        plan.total_farm_area_ha = parcel_result["area"]

        plan.soil_type = soil_data_result.get("soil_type", "Not available.")
        plan.nutrient_leaching_vulnerability = soil_data_result.get("nutrient_leaching_vulnerability")
        plan.erodibility = soil_data_result.get("erodibility")

        try:
            # Ensure the value from ArcGIS is not None before trying to convert to float
            arcgis_slope = soil_data_result.get("arcgis_slope_angle")
            if arcgis_slope is not None:
                plan.arcgis_slope_angle = float(arcgis_slope)
            else:
                plan.arcgis_slope_angle = slope_angle_dem
        except (ValueError, TypeError):
            plan.arcgis_slope_angle = slope_angle_dem

        plan.soil_drainage_class = soil_drainage_result
        plan.slope_class = slope_class_result

        _update_progress(plan.pk, "Regional environmental data retrieved.", "complete")

        plan.generation_status = FreshwaterPlan.GenerationStatus.READY
        plan.save()
        logger.info(f"populate_admin_details_task completed for plan ID: {plan_id}")

    except FreshwaterPlan.DoesNotExist:
        logger.warning(f"process_fetched_data_task: plan ID {plan_id} does not exist")
    except Exception as e:
        logger.error(f"process_fetched_data_task failed for ID {plan_id}: {e}", exc_info=True)
        _update_progress(plan_id, f"A critical error occurred while processing data: {e}", "error")
        plan.generation_status = FreshwaterPlan.GenerationStatus.FAILED
        plan.save(update_fields=['generation_status', 'updated_at'])


@job('default')
def populate_admin_details_task(freshwater_plan_id):
    """
    Orchestrates fetching administrative and environmental details.
    """
    try:
        logger.info(f"PlanID={freshwater_plan_id} - populate_admin_details_task: Task started.")
        plan = FreshwaterPlan.objects.get(pk=freshwater_plan_id)
        queue = django_rq.get_queue('high')

        _update_progress(plan.pk, f"Verifying council authority for {plan.council} area...", "pending")
        _update_progress(plan.pk, "The full plan will involve a risk analysis based on catchment data...", "info")
        _update_progress(plan.pk, "Fetching address, property, and environmental data...", "pending")

        # Enqueue data fetching tasks
        council_name_job = queue.enqueue(get_council_authority_name_task, plan.council) # New task
        address_job = queue.enqueue(get_address_task, plan.latitude, plan.longitude)
        catchment_job = queue.enqueue(get_catchment_task, plan.longitude, plan.latitude)
        parcel_job = queue.enqueue(get_parcel_task, plan.longitude, plan.latitude)
        soil_data_job = queue.enqueue(get_soil_data_task, plan.longitude, plan.latitude)
        soil_drainage_job = queue.enqueue(get_soil_drainage_task, plan.latitude, plan.longitude)
        slope_class_job = queue.enqueue(get_slope_class_task, plan.longitude, plan.latitude)

        # Enqueue the processing task, making it dependent on the data fetching jobs
        processing_job = django_rq.get_queue('default').enqueue(
            process_fetched_data_task,
            plan.pk,
            depends_on=[council_name_job, address_job, catchment_job, parcel_job, soil_data_job, soil_drainage_job, slope_class_job]
        )

    except FreshwaterPlan.DoesNotExist:
        logger.warning(f"populate_admin_details_task: plan ID {freshwater_plan_id} does not exist")
    except Exception as e:
        logger.error(f"populate_admin_details_task failed for ID {freshwater_plan_id}: {e}", exc_info=True)
        _update_progress(freshwater_plan_id, f"A critical error occurred: {e}", "error")
        plan.generation_status = FreshwaterPlan.GenerationStatus.FAILED
        plan.save(update_fields=['generation_status', 'updated_at'])
        raise


@job('default')
def get_es_water_quality_for_site(site_pk, measurement, user_lat, user_lon, distance_km):
    """
    An RQ task to fetch water quality data for a single site and measurement.
    """
    try:
        # Import locally to prevent circular dependency: views -> tasks -> models
        from .models import WaterQualityLog
        from .services.data_service import DataService
        site = MonitoringSite.objects.get(pk=site_pk)
        data_service = DataService()

        result = data_service.get_water_quality_data([site], measurement, user_lat, user_lon)

        if result and not result.get('mock_data'):
            log_entry_data = {
                'site_name': site.site_name,
                'latitude': site.latitude,
                'longitude': site.longitude,
                'distance_to_user_point_km': distance_km,
                'user_point_lat': user_lat,
                'user_point_lon': user_lon,
                'source_api': result.get('source'),
            }

            measurement_field_map = {
                "E-Coli <CFU>": "e_coli",
                "Nitrogen (Nitrate Nitrite)": "nitrate",
                "Turbidity (FNU)": "turbidity",
            }
            model_field = measurement_field_map.get(measurement)
            if model_field:
                log_entry_data[model_field] = result.get('average_value')

            WaterQualityLog.objects.create(**log_entry_data)
            return result

        return None

    except MonitoringSite.DoesNotExist:
        logger.error(f"Task failed: MonitoringSite with PK {site_pk} does not exist.")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in get_es_water_quality_for_site for site {site_pk}: {e}", exc_info=True)
        return None


# --- LAWA Water Quality Data Fetching Task ---

def _haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the distance between two points in kilometers."""
    R = 6371  # Earth radius in kilometers
    
    # Ensure coordinates are valid floats
    try:
        lat1, lon1, lat2, lon2 = map(math.radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
    except (ValueError, TypeError):
        return float('inf') # Return infinity if coordinates are invalid

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c

def _query_southland_beacon_fallback(lat, lon):
    """
    Queries the Southland 'Beacon' identify service as a fallback.
    """
    logger.info(f"Primary LAWA search failed for Southland coordinates. Attempting Beacon fallback.")
    try:
        # Approximate conversion from WGS84 to NZTM 2193
        x_nztm = 1200000 + lon * 40000
        y_nztm = 5000000 - lat * 40000

        url = "https://maps.es.govt.nz/server/rest/services/Public/WaterAndLand/MapServer/identify"
        params = {
            "geometry": f"{x_nztm},{y_nztm}",
            "geometryType": "esriGeometryPoint",
            "sr": "2193",
            "layers": "visible:27",
            "tolerance": "20",
            "mapExtent": f"{x_nztm-100},{y_nztm-100},{x_nztm+100},{y_nztm+100}",
            "imageDisplay": "400,300,96",
            "returnGeometry": "false",
            "f": "json"
        }
        
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()

        if not data.get("results"):
            logger.warning("Southland Beacon fallback returned no results.")
            return None

        attributes = data["results"][0].get("attributes", {})
        site_name = data["results"][0].get("value", "Unknown Beacon Site")

        measurement_map = {
            "Degradation_Ecoli": "E-Coli <CFU>",
            "Degradation_Suspended_Sediment": "Turbidity (FNU)",
            "Degradation_Nitrogen": "Nitrogen (Nitrate Nitrite)",
            "Degradation_Phosphorus": "Phosphorus (Dissolved Reactive)"
        }
        
        report_data = {}
        for beacon_field, measurement_name in measurement_map.items():
            if beacon_field in attributes:
                report_data[measurement_name] = {
                    'status_value': attributes[beacon_field],
                    'data_points': 1,
                    'source': 'Beacon: Degradation Status'
                }

        if not report_data:
            logger.warning("Southland Beacon fallback result had no mappable degradation fields.")
            return None

        return {
            'site_name': site_name,
            'distance_km': 0,
            'data': report_data
        }

    except requests.RequestException as e:
        logger.error(f"Southland Beacon fallback query failed: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred in Southland Beacon fallback: {e}", exc_info=True)
        return None

@job('default', result_ttl=3600)
def fetch_lawa_water_quality_task(user_lat, user_lon, measurements):
    """
    Finds the nearest LAWA site with valid data and fetches water quality measurements.
    It will iterate through sites by proximity until data is found.
    """
    # 1. Load cached LAWA sites
    try:
        sites_path = os.path.join(settings.BASE_DIR, 'doc_generator', 'data', 'lawa_sites.json')
        with open(sites_path, 'r') as f:
            lawa_sites_data = json.load(f)
        lawa_sites = lawa_sites_data.get("sites", [])
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Could not load lawa_sites.json: {e}. Cannot perform LAWA lookup.")
        return {"error": "LAWA site cache is missing or corrupt."}

    # 2. Calculate distance to all sites and sort them
    sites_with_distance = []
    for site in lawa_sites:
        distance = _haversine_distance(user_lat, user_lon, site['Lat'], site['Long'])
        sites_with_distance.append((site, distance))
    
    sites_with_distance.sort(key=lambda x: x[1])

    # 3. Iterate through sites and query LAWA API
    DISTANCE_THRESHOLD_KM = 50
    SITES_TO_CHECK_LIMIT = 5

    for site_info, distance in sites_with_distance[:SITES_TO_CHECK_LIMIT]:
        if distance > DISTANCE_THRESHOLD_KM:
            logger.info(f"Next closest site is beyond the {DISTANCE_THRESHOLD_KM}km threshold. Stopping search.")
            break

        logger.info(f"Attempting to fetch data from site '{site_info['SiteName']}' at {distance:.2f} km.")
        
        measurement_map = {
            'Turbidity (FNU)': 'TURB',
            'E-Coli <CFU>': 'ECOLI',
            'Nitrogen (Nitrate Nitrite)': 'NO3N',
            'Phosphorus (Dissolved Reactive)': 'DRP',
        }
        
        lawa_results = {}
        current_year = time.strftime("%Y")

        for measurement_name in measurements:
            indicator = measurement_map.get(measurement_name)
            if not indicator:
                continue

            try:
                lawa_url = f"https://www.lawa.org.nz/umbraco/api/riverservice/GetRecGraphData"
                params = {
                    'location': site_info['LawaId'],
                    'indicator': indicator,
                    'startYear': '2015',
                    'endYear': current_year
                }
                response = requests.get(lawa_url, params=params, timeout=20)
                response.raise_for_status()
                nested_data_str = response.json().get('data')
                if not nested_data_str:
                    data = []
                elif isinstance(nested_data_str, str):
                    try:
                        data = json.loads(nested_data_str)
                    except json.JSONDecodeError:
                        data = []
                elif isinstance(nested_data_str, list):
                    data = nested_data_str
                else:
                    data = []
                
                values = [d[1] for d in data if d and len(d) > 1 and d[1] is not None]
                if values:
                    average_value = sum(values) / len(values)
                    lawa_results[measurement_name] = {
                        'average_value': round(average_value, 2),
                        'data_points': len(values),
                        'source': 'LAWA'
                    }
            except requests.RequestException as e:
                logger.warning(f"LAWA API query failed for site {site_info['LawaId']} and indicator {indicator}: {e}.")
        
        if lawa_results:
            logger.info(f"Successfully fetched data from site '{site_info['SiteName']}'.")
            return {
                'site_name': site_info['SiteName'],
                'distance_km': round(distance, 2),
                'data': lawa_results
            }
        else:
            logger.info(f"No data found for site '{site_info['SiteName']}'. Trying next closest site.")

    logger.warning(f"No suitable LAWA site with *valid data* found within {DISTANCE_THRESHOLD_KM}km after checking {SITES_TO_CHECK_LIMIT} sites.")
    
    # Southland Beacon Fallback
    if -47 < user_lat < -45 and 167 < user_lon < 169:
        fallback_result = _query_southland_beacon_fallback(user_lat, user_lon)
        if fallback_result:
            return fallback_result

    return {
        "status": "no_data",
        "message": f"No water quality data could be found from the {SITES_TO_CHECK_LIMIT} nearest monitoring sites within {DISTANCE_THRESHOLD_KM}km.",
        "data": {}
    }

@job('default', result_ttl=3600)
def fetch_lawa_data_for_closest_sites_task(user_lat, user_lon):
    """
    Finds the 5 nearest LAWA sites that have measurement data and retrieves it.
    """
    logger.info("--- RUNNING LATEST VERSION OF FETCH TASK ---")
    logger.info(f"Starting LAWA data fetch for 5 closest sites with data near ({user_lat}, {user_lon})")
    
    try:
        site_index_path = os.path.join(settings.BASE_DIR, 'doc_generator', 'data', 'lawa_sites.json')
        with open(site_index_path, 'r') as f:
            all_sites_data = json.load(f)
        all_sites = all_sites_data.get("sites", [])

        measurements_path = os.path.join(settings.BASE_DIR, 'doc_generator', 'data', 'lawa_measurements.json')
        with open(measurements_path, 'r') as f:
            all_measurements = json.load(f)
            
    except (FileNotFoundError, json.JSONDecodeError) as e:
        error_msg = f"Could not load pre-processed LAWA data files: {e}"
        logger.error(error_msg)
        return {"error": error_msg}

    sites_with_distance = []
    for site in all_sites:
        if site.get('Lat') is not None and site.get('Long') is not None:
            distance = _haversine_distance(user_lat, user_lon, site['Lat'], site['Long'])
            sites_with_distance.append({'site_info': site, 'distance': distance})
    
    sites_with_distance.sort(key=lambda x: x['distance'])

    # New logic: Find up to 5 sites that HAVE measurements.
    closest_sites_with_data = []
    for item in sites_with_distance:
        if len(closest_sites_with_data) >= 5:
            break # Stop once we have 5 sites

        site_id = item['site_info']['LawaId']
        measurements = all_measurements.get(site_id, [])
        
        if measurements: # Only add the site if it has measurement data
            closest_sites_with_data.append({
                'lawa_id': site_id, # Add LawaId for the frontend
                'site_name': item['site_info']['SiteName'],
                'distance': item['distance'],
                'measurements': measurements
            })

    logger.info(f"Successfully retrieved data for {len(closest_sites_with_data)} closest sites with measurements.")
    return closest_sites_with_data

@job('default', result_ttl=3600)
def analyze_lawa_data_task(*, plan_pk):
    """
    Task 2 in the water quality chain. Takes raw LAWA data, generates an AI summary,
    and returns the final combined report.
    """
    current_job = get_current_job()
    if not current_job:
        error_msg = f"PlanID={plan_pk} - analyze_lawa_data_task could not access current job context."
        logger.error(error_msg)
        return {"sites_data": [], "ai_summary": "Analysis skipped: No job context available."}

    # Use dependency IDs for compatibility with RQ versions that don't expose `dependencies`
    dep_ids = getattr(current_job, "dependency_ids", None) or []
    if not dep_ids:
        error_msg = f"PlanID={plan_pk} - analyze_lawa_data_task was called without a dependency. Cannot fetch previous result."
        logger.error(error_msg)
        return {"sites_data": [], "ai_summary": "Analysis skipped: No dependency result available."}

    # Fetch the result from the dependency (the fetch_lawa_data_for_closest_sites_task job)
    try:
        dependency_job = Job.fetch(dep_ids[0], connection=current_job.connection)
    except Exception as e:
        logger.error(f"PlanID={plan_pk} - Failed to fetch dependency job {dep_ids[0]}: {e}", exc_info=True)
        return {"sites_data": [], "ai_summary": "Analysis skipped: Failed to load dependency result."}

    previous_result = dependency_job.result
    if getattr(dependency_job, "is_failed", False) or previous_result is None:
        logger.warning(f"PlanID={plan_pk} - Dependency job {getattr(dependency_job, 'id', dep_ids[0])} failed or returned a None result. Aborting analysis.")
        return {"sites_data": [], "ai_summary": "Analysis skipped: Could not retrieve water quality data."}

    logger.info(f"PlanID={plan_pk} - Starting AI analysis of LAWA data.")

    if isinstance(previous_result, dict) and previous_result.get('error'):
        logger.warning(f"PlanID={plan_pk} - Skipping AI analysis due to an error reported in the previous step: {previous_result.get('error')}")
        return {"sites_data": previous_result, "ai_summary": "Analysis skipped: An error occurred while fetching water quality data."}

    if not previous_result: # This specifically handles the empty list case now
        logger.warning(f"PlanID={plan_pk} - The data fetching step returned no sites. Proceeding to generate a report indicating this.")
        return {"sites_data": [], "ai_summary": "No nearby monitoring sites were found, so a detailed water quality analysis could not be performed."}

    try:
        plan = FreshwaterPlan.objects.get(pk=plan_pk)
        site_context = {"pk": plan.pk, "council_authority_name": plan.council_authority_name}

        # This method does not exist on VulnerabilityService. The logic should be self-contained
        # or call a valid method. For now, we'll create a placeholder summary.
        # A proper implementation would involve a call to an LLM.
        ai_summary, error = ("AI summary generation is not yet fully implemented.", None)

        if error:
            logger.error(f"PlanID={plan_pk} - Error during LAWA AI analysis: {error}")

        final_report = {
            "sites_data": previous_result,
            "ai_summary": ai_summary
        }
        return final_report

    except Exception as e:
        logger.error(f"PlanID={plan_pk} - A critical error occurred in analyze_lawa_data_task: {e}", exc_info=True)
        return {"sites_data": previous_result if 'previous_result' in locals() else [], "ai_summary": "A critical error prevented AI analysis."}


@job('default')
def generate_water_quality_report_task(plan_pk, **kwargs):
    """
    Orchestrator task that chains the data fetching and analysis for the water quality report.
    """
    logger.info(f"PlanID={plan_pk} - Orchestrator task started.")
    try:
        logger.info(f"PlanID={plan_pk} - Fetching FreshwaterPlan object.")
        plan = FreshwaterPlan.objects.get(pk=plan_pk)
        logger.info(f"PlanID={plan_pk} - Successfully fetched plan.")

        final_job_id = kwargs.get('final_job_id') or f"water_quality_report_{plan_pk}"
        fetch_job_id = f"fetch_data_for_{final_job_id}"
        logger.info(f"PlanID={plan_pk} - Final Job ID: {final_job_id}, Fetch Job ID: {fetch_job_id}")

        queue = django_rq.get_queue('default')
        logger.info(f"PlanID={plan_pk} - Enqueuing fetch task: fetch_lawa_data_for_closest_sites_task")
        
        fetch_job = queue.enqueue(
            fetch_lawa_data_for_closest_sites_task, 
            plan.latitude, 
            plan.longitude, 
            job_id=fetch_job_id
        )
        # CRITICAL: Log the ID of the job we just created.
        logger.info(f"PlanID={plan_pk} - Fetch task enqueued with Job ID: {fetch_job.id}")

        logger.info(f"PlanID={plan_pk} - Enqueuing analysis task (analyze_lawa_data_task) to depend on {fetch_job.id}")
        analysis_job = queue.enqueue(
            analyze_lawa_data_task, 
            depends_on=fetch_job, 
            job_id=final_job_id, 
            kwargs={'plan_pk': plan_pk}
        )
        logger.info(f"PlanID={plan_pk} - Analysis task enqueued with Job ID: {analysis_job.id}")
        logger.info(f"PlanID={plan_pk} - Orchestrator task finished successfully.")

    except FreshwaterPlan.DoesNotExist:
        logger.error(f"PlanID={plan_pk} - Orchestrator failed: FreshwaterPlan with pk={plan_pk} does not exist.")
    except Exception as e:
        logger.error(f"PlanID={plan_pk} - Orchestrator failed with an unexpected error: {e}", exc_info=True)
        raise

def _analyze_water_quality_with_rag(water_quality_results: dict, council_name: str) -> str:
    """
    Uses a RAG pipeline to generate an expert analysis of water quality data.

    Args:
        water_quality_results: The dictionary containing the fetched water quality data.
        council_name: The name of the regional council for context retrieval.

    Returns:
        A string containing the AI-generated analysis, or an empty string on failure.
    """
    logger.info("Starting RAG analysis of water quality data.")
    try:
        # 1. Initialize RAG components
        embeddings = get_embedding_model()
        vector_store_path = os.path.join(settings.BASE_DIR, 'vector_store')
        vector_store = Chroma(persist_directory=vector_store_path, embedding_function=embeddings)
        retriever = vector_store.as_retriever(search_kwargs={'k': 5})
        llm = OllamaLLM(model="phi3:mini", base_url="http://localhost:11434")

        # 2. Define the expert prompt
        template = """
        You are a senior environmental consultant in New Zealand. Your task is to provide a holistic analysis of the following water quality data.
        The monitoring site is located within the '{council}' regional council area.

        **Retrieved Regulatory Context:**
        {context}

        **Water Quality Data:**
        {water_data}

        ---
        **Analysis Task:**
        Based on the provided data and regulatory context, write a concise paragraph that explains what these results mean for freshwater quality.
        Adopt a holistic perspective, considering the implications of these values in the context of New Zealand's freshwater legislation and regional policies.
        Explain potential risks or concerns indicated by the data (e.g., high turbidity suggesting sediment runoff, high E. coli indicating contamination).
        Your analysis should be clear, authoritative, and easy for a landowner to understand.
        """
        prompt = PromptTemplate.from_template(template)

        # 3. Construct and invoke the RAG chain
        rag_chain = ({"context": retriever, "council": lambda x: council_name, "water_data": lambda x: json.dumps(water_quality_results, indent=2)} | prompt | llm | StrOutputParser())
        return rag_chain.invoke(f"Analyze water quality data for {council_name}")
    except Exception as e:
        logger.error(f"Failed to generate RAG analysis for water quality data: {e}", exc_info=True)
        return "AI analysis could not be generated at this time due to a system error."

# --- New Chained Vulnerability Analysis Tasks ---

@job('default')
def run_retrieval_step(*, plan_pk, final_job_id=None):
    logger.info(f"run_retrieval_step started for plan_pk={plan_pk}, final_job_id={final_job_id}")
    current_job = get_current_job() # Get the current job instance
    try:
        # Imports are placed inside the task to avoid potential top-level circular dependencies.
        from .models import FreshwaterPlan
        from langchain_chroma import Chroma
        from .services.embedding_service import get_embedding_model
        import os
        from django.conf import settings
        from .services.vulnerability_service import VulnerabilityService

        plan = FreshwaterPlan.objects.get(pk=plan_pk)

        # Initialize the components required by the VulnerabilityService
        embeddings = get_embedding_model()
        vector_store_path = os.path.join(settings.BASE_DIR, 'vector_store')
        vector_store = Chroma(persist_directory=vector_store_path, embedding_function=embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 10})

        # The service now expects a dictionary of site context, not the plan object.
        site_context = {
            "pk": plan.pk,
            "council_authority_name": plan.council_authority_name,
            "catchment_name": plan.catchment_name,
            "soil_type": plan.soil_type,
            "arcgis_slope_angle": plan.arcgis_slope_angle,
            "erodibility": plan.erodibility,
            "soil_drainage_class": plan.soil_drainage_class,
        }

        # Instantiate the service with the correct arguments
        service = VulnerabilityService(retriever=retriever, site_context=site_context)
        
        # The _fetch_and_build_site_context method is no longer needed as context is passed on init.
        # Directly call the retrieval method.
        retrieval_result, error = service._perform_advanced_retrieval()
        logger.info(f"PlanID={plan_pk} - _perform_advanced_retrieval completed. Error: {error}")

        if error:
            logger.error(f"PlanID={plan_pk} - Retrieval step failed with error: {error}")
            raise Exception(f"Retrieval step failed: {error}")
            
        final_result = {
            "combined_context": retrieval_result.get("combined_context", ""),
            "docs": retrieval_result.get("docs", []),
            "site_context": service.site_context
        }
        logger.info(f"PlanID={plan_pk} - run_retrieval_step completed. Enqueuing summarization step.")

        # Enqueue the next job in the chain
        queue = django_rq.get_queue('default')
        final_job_id = current_job.meta.get('final_job_id') # Retrieve final_job_id from meta
        summarization_job = queue.enqueue(
            run_summarization_step,
            final_result, # Pass the result of this job
            plan_pk=plan_pk,
            final_job_id=final_job_id, # Pass final_job_id
            depends_on=current_job, # Ensure order
            result_ttl=3600
        )
        summarization_job.meta['plan_pk'] = plan_pk
        summarization_job.meta['final_job_id'] = final_job_id
        summarization_job.save_meta()
        logger.info(f"Enqueued summarization_job (ID: {summarization_job.id}) for plan_pk={plan_pk}")

        return final_result # Still return the result for potential inspection, though not used by on_success anymore

    except Exception as e:
        logger.error(f"PlanID={plan_pk} - An unexpected error occurred in run_retrieval_step: {e}", exc_info=True)
        raise


@job('default')
def run_summarization_step(previous_result=None, *, plan_pk, final_job_id=None):
    logger.info(f"run_summarization_step started for plan_pk={plan_pk}, final_job_id={final_job_id}")
    current_job = get_current_job() # Get the current job instance
    try:
        if previous_result is None:
            logger.warning(f"PlanID={plan_pk} - No previous_result provided for summarization step. Proceeding with empty context.")
            previous_result = {}

        site_context = previous_result.get("site_context", {})
        combined_context = previous_result.get("combined_context", "")

        service = _get_vulnerability_service(plan_pk, site_context)
        summarized_regulatory_context, reg_error = service._summarize_retrieved_context(docs=previous_result.get("docs", []), combined_context=combined_context, site_context=site_context)
        catchment_summary, catchment_error = service.summarize_catchment_context(combined_context=combined_context, site_context=site_context)

        if reg_error or catchment_error:
            logger.warning(f"PlanID={plan_pk} - Errors in summarization: REG_ERR: {reg_error}, CATCH_ERR: {catchment_error}")

        step_result = {
            "combined_context": combined_context,
            "summarized_regulatory_context": summarized_regulatory_context,
            "catchment_summary": catchment_summary, # This is now generated statically
            "site_context": site_context # Pass the context through
        }
        logger.info(f"PlanID={plan_pk} - run_summarization_step completed. Enqueuing risk identification step.")

        # Enqueue the next job in the chain
        queue = django_rq.get_queue('default')
        risk_job = queue.enqueue(
            run_risk_identification_step,
            step_result, # Pass the result of this job
            plan_pk=plan_pk,
            final_job_id=final_job_id, # Pass final_job_id
            depends_on=current_job, # Ensure order
            result_ttl=3600
        )
        risk_job.meta['plan_pk'] = plan_pk
        risk_job.meta['final_job_id'] = final_job_id
        risk_job.save_meta()
        logger.info(f"Enqueued risk_job (ID: {risk_job.id}) for plan_pk={plan_pk}")

        return step_result

    except Exception as e:
        logger.error(f"PlanID={plan_pk} - An unexpected error occurred in run_summarization_step: {e}", exc_info=True)
        raise


@job('default')
def run_risk_identification_step(previous_result=None, *, plan_pk, final_job_id=None):
    logger.info(f"run_risk_identification_step started for plan_pk={plan_pk}, final_job_id={final_job_id}")
    current_job = get_current_job() # Get the current job instance
    try:
        if previous_result is None:
            logger.warning(f"PlanID={plan_pk} - No previous_result provided for risk identification step. Proceeding with empty context.")
            previous_result = {}

        site_context = previous_result.get("site_context", {})
        
        service = _get_vulnerability_service(plan_pk, site_context)
        identified_risks, risk_error = service.identify_risks_from_data(site_context=site_context)
        if risk_error:
            logger.warning(f"PlanID={plan_pk} - Error in risk identification: {risk_error}")

        step_result = previous_result # Start with previous result
        step_result["identified_risks"] = identified_risks # Add identified risks
        logger.info(f"PlanID={plan_pk} - run_risk_identification_step completed. Enqueuing final analysis step.")

        # Enqueue the next job in the chain
        queue = django_rq.get_queue('default')
        final_analysis_job = queue.enqueue(
            run_final_analysis_step,
            step_result, # Pass the result of this job
            plan_pk=plan_pk,
            final_job_id=final_job_id, # Pass final_job_id
            job_id=final_job_id, # Use the predictable ID for the final job
            depends_on=current_job,
            result_ttl=3600
        )
        final_analysis_job.meta['plan_pk'] = plan_pk
        final_analysis_job.meta['final_job_id'] = final_job_id
        final_analysis_job.save_meta()
        logger.info(f"Enqueued final_analysis_job (ID: {final_analysis_job.id}) for plan_pk={plan_pk}")

        return step_result

    except Exception as e:
        logger.error(f"PlanID={plan_pk} - An unexpected error occurred in run_risk_identification_step: {e}", exc_info=True)
        raise


@job('default')
def run_final_analysis_step(previous_result=None, *, plan_pk, final_job_id=None):
    logger.info(f"run_final_analysis_step started for plan_pk={plan_pk}, final_job_id={final_job_id}")
    current_job = get_current_job() # Get the current job instance
    try:
        if previous_result is None:
            logger.warning(f"PlanID={plan_pk} - No previous_result provided for final analysis step. Proceeding with empty context.")
            previous_result = {}

        site_context = previous_result.get("site_context", {})

        structured_report, errors = VulnerabilityService.generate_structured_analysis(
            catchment_summary=previous_result.get("catchment_summary"),
            identified_risks=previous_result.get("identified_risks"),
            regulatory_context=previous_result.get("summarized_regulatory_context"),
            site_context=site_context
        )
        if errors:
            logger.warning(f"PlanID={plan_pk} - Errors encountered during final analysis: {errors}")

        plan = FreshwaterPlan.objects.get(pk=plan_pk)
        final_data = {
            'status': 'finished', # Use 'finished' to match frontend expectation
            'progress': 'Analysis complete',
            'data': structured_report
        }
        plan.vulnerability_analysis_data = final_data
        plan.save(update_fields=['vulnerability_analysis_data', 'updated_at'])

        logger.info(f"Successfully generated and saved final analysis report for plan_pk={plan_pk}")
        return structured_report # Return the report directly, not the entire saved object

    except Exception as e:
        logger.error(f"PlanID={plan_pk} - An unexpected error occurred in run_final_analysis_step: {e}", exc_info=True)
        raise





@job('default')
def run_vulnerability_analysis_v2_task(plan_pk, final_job_id=None):
    """
    Orchestrates the vulnerability analysis by chaining the individual steps.
    """
    logger.info(f"Starting vulnerability analysis for plan_pk={plan_pk}")
    plan = None
    try:
        plan = FreshwaterPlan.objects.get(pk=plan_pk)
        queue = django_rq.get_queue('default')

        # Enqueue the first job in the chain
        retrieval_job = queue.enqueue(
            run_retrieval_step,
            plan_pk=plan_pk,
            final_job_id=final_job_id, # Pass final_job_id
            result_ttl=3600
        )
        retrieval_job.meta['plan_pk'] = plan_pk
        retrieval_job.meta['final_job_id'] = final_job_id
        retrieval_job.save_meta()
        logger.info(f"Enqueued retrieval_job (ID: {retrieval_job.id}) for plan_pk={plan_pk}")

    except FreshwaterPlan.DoesNotExist:
        logger.error(f"run_vulnerability_analysis_v2_task failed: Plan with pk={plan_pk} not found.")
    except Exception as e:
        logger.error(f"A critical error occurred in run_vulnerability_analysis_v2_task for plan_pk={plan_pk}: {e}", exc_info=True)
        if plan:
            plan.vulnerability_analysis_data = {"errors": [f"A critical background task error occurred: {str(e)}"]}
            plan.save(update_fields=['vulnerability_analysis_data'])
        raise