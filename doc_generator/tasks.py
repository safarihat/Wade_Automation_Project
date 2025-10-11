import os
import io
import logging
import time
import requests
import django_rq
from django.conf import settings
from django.db import transaction
from django.template.loader import render_to_string
from django.core.files.base import ContentFile
from doc_generator.models import FreshwaterPlan
from doc_generator.geospatial_utils import (
    transform_coords,
    _query_koordinates_vector,
    _query_arcgis_vector,
    _query_koordinates_raster,
    _query_arcgis_raster,
    _calculate_slope_from_dem,
)
from doc_generator.services.soil_drainage_service import SoilDrainageService
from doc_generator.services.data_service import DataService
from doc_generator.services.vulnerability_service import VulnerabilityService
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
    try:
        embeddings = get_embedding_model()
        vector_store_path = os.path.join(settings.BASE_DIR, 'vector_store')
        vector_store = Chroma(persist_directory=vector_store_path, embedding_function=embeddings)
        retriever = vector_store.as_retriever(search_kwargs={'k': 1})
        llm = OllamaLLM(model="phi3:mini", base_url="http://localhost:11434", options={"num_ctx": 2048})

        template = """Based on the context for the '{council}' area, what is the official name of the regional council or unitary authority? Provide only the name.
        Context: {context}
        Answer:"""
        prompt = PromptTemplate.from_template(template)
        rag_chain = ({"context": retriever, "council": lambda x: council_name} | prompt | llm | StrOutputParser())
        return rag_chain.invoke(f"Official name for {council_name}")
    except Exception as e:
        logger.warning(f"RAG query for council authority name failed for council {council_name}: {e}")
        return f"Could not verify '{council_name}'"


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
            "request": "GetFeature",
            "typeNames": "layer-53682",
            "outputFormat": "application/json",
            "srsName": "EPSG:4326",
            "cql_filter": f"INTERSECTS(geometry,POINT({lon} {lat}))"
        }
        response = requests.get(wfs_url, params=params, timeout=15)
        response.raise_for_status()
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


# --- New Chained Vulnerability Analysis Tasks ---

@job('default')
def run_retrieval_step(plan_pk):
    logger.info(f"PlanID={plan_pk} - VULN_ANALYSIS [1/4]: Running Retrieval Step")
    plan = FreshwaterPlan.objects.get(pk=plan_pk)
    service = VulnerabilityService(plan=plan)
    service._fetch_and_build_site_context()
    retrieval_result, error = service._perform_advanced_retrieval()
    if error:
        raise Exception(f"Retrieval step failed: {error}")
    return {"combined_context": retrieval_result.get("combined_context", ""),
            "site_context": service.site_context}


@job('default')
def run_summarization_step(previous_result, plan_pk):
    logger.info(f"PlanID={plan_pk} - VULN_ANALYSIS [2/4]: Running Summarization Step")
    plan = FreshwaterPlan.objects.get(pk=plan_pk)
    service = VulnerabilityService(plan=plan)
    combined_context = previous_result.get("combined_context", "")
    service.site_context = previous_result.get("site_context", {})

    summarized_regulatory_context, reg_error = service._summarize_retrieved_context(combined_context=combined_context)
    catchment_summary, catchment_error = service.summarize_catchment_context(combined_context)

    if reg_error or catchment_error:
        logger.warning(f"PlanID={plan_pk} - Errors in summarization: REG_ERR: {reg_error}, CATCH_ERR: {catchment_error}")

    plan.vulnerability_analysis_data = {
        "status": "processing",
        "progress": "Summarization complete",
        "data": {
            "summarized_regulatory_context": summarized_regulatory_context,
            "catchment_context_summary": catchment_summary,
        }
    }
    plan.save(update_fields=['vulnerability_analysis_data'])

    return {
        "combined_context": combined_context,
        "summarized_regulatory_context": summarized_regulatory_context,
        "catchment_summary": catchment_summary,
        "site_context": service.site_context
    }


@job('default')
def run_risk_identification_step(previous_result, plan_pk):
    logger.info(f"PlanID={plan_pk} - VULN_ANALYSIS [3/4]: Running Risk Identification Step")
    plan = FreshwaterPlan.objects.get(pk=plan_pk)
    service = VulnerabilityService(plan=plan)
    service.site_context = previous_result.get("site_context", {})

    identified_risks, risk_error = service.identify_risks_from_data()
    if risk_error:
        logger.warning(f"PlanID={plan_pk} - Error in risk identification: {risk_error}")

    current_data = plan.vulnerability_analysis_data or {"data": {}}
    current_data["progress"] = "Risk identification complete"
    current_data["data"]["identified_risks"] = identified_risks
    plan.vulnerability_analysis_data = current_data
    plan.save(update_fields=['vulnerability_analysis_data'])

    previous_result["identified_risks"] = identified_risks
    return previous_result


@job('default')
def run_final_analysis_step(previous_result, plan_pk):
    logger.info(f"PlanID={plan_pk} - VULN_ANALYSIS [4/4]: Running Final Analysis Step")
    plan = FreshwaterPlan.objects.get(pk=plan_pk)
    service = VulnerabilityService(plan=plan)
    service.site_context = previous_result.get("site_context", {})

    structured_report, errors = service.generate_structured_analysis(
        catchment_summary=previous_result.get("catchment_summary"),
        identified_risks=previous_result.get("identified_risks"),
        regulatory_context=previous_result.get("summarized_regulatory_context")
    )

    final_data = plan.vulnerability_analysis_data or {"data": {}}
    final_data["data"].update(structured_report)
    final_data["progress"] = "Analysis complete"
    plan.vulnerability_analysis_data = final_data
    plan.vulnerability_analysis_status = FreshwaterPlan.GenerationStatus.READY
    plan.save(update_fields=['vulnerability_analysis_data', 'vulnerability_analysis_status'])
    logger.info(f"Successfully completed vulnerability analysis for plan_pk={plan_pk}")


@job('default')
def run_vulnerability_analysis_task(plan_pk):
    """
    Orchestrates the vulnerability analysis by chaining the individual steps.
    """
    logger.info(f"Starting vulnerability analysis for plan_pk={plan_pk}")
    plan = None
    try:
        plan = FreshwaterPlan.objects.get(pk=plan_pk)
        queue = django_rq.get_queue('default')

        # Enqueue jobs with dependencies
        retrieval_job = queue.enqueue(run_retrieval_step, plan_pk)
        summarization_job = queue.enqueue(run_summarization_step, plan_pk, depends_on=retrieval_job, result_ttl=3600)
        risk_job = queue.enqueue(run_risk_identification_step, plan_pk, depends_on=summarization_job, result_ttl=3600)
        final_job = queue.enqueue(run_final_analysis_step, plan_pk, depends_on=risk_job, result_ttl=3600)

    except FreshwaterPlan.DoesNotExist:
        logger.error(f"run_vulnerability_analysis_task failed: Plan with pk={plan_pk} not found.")
    except Exception as e:
        logger.error(f"A critical error occurred in run_vulnerability_analysis_task for plan_pk={plan_pk}: {e}", exc_info=True)
        if plan:
            plan.vulnerability_analysis_status = FreshwaterPlan.GenerationStatus.FAILED
            plan.vulnerability_analysis_data = {"errors": [f"A critical background task error occurred: {str(e)}"]}
            plan.save(update_fields=['vulnerability_analysis_status', 'vulnerability_analysis_data'])
        raise