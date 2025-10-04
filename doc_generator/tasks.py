import os
import io
import logging
import requests
import gevent
from celery import shared_task
from django.conf import settings
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
import hashlib
from langchain_core.documents import Document

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
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from .services.embedding_service import get_embedding_model

# Setup logger
logger = logging.getLogger(__name__)

def _create_watermark_pdf() -> io.BytesIO:
    """Creates a PDF in memory containing only the 'PREVIEW ONLY' watermark."""
    packet = io.BytesIO()
    # Create a new PDF with Reportlab
    can = canvas.Canvas(packet, pagesize=letter)
    can.saveState()
    can.setFont('Helvetica-Bold', 60)
    can.setFillColor(colors.grey, alpha=0.2)
    # Center and rotate the watermark text
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
    
    # Read the main PDF content and the watermark
    main_pdf = PdfReader(io.BytesIO(pdf_content))
    watermark_page = PdfReader(watermark_pdf).pages[0]

    # Iterate over all pages of the main PDF and merge the watermark
    for page in main_pdf.pages:
        page.merge_page(watermark_page)
        writer.add_page(page)

    writer.write(output_pdf_stream)
    output_pdf_stream.seek(0)
    return output_pdf_stream

@shared_task(bind=True, max_retries=3, default_retry_delay=60, soft_time_limit=900, time_limit=960)
def generate_plan_task(self, freshwater_plan_id):
    """
    Celery task to generate a freshwater plan using a RAG pipeline,
    then create a watermarked PDF preview.
    """
    logger.info(f"--- Starting plan generation for FreshwaterPlan ID: {freshwater_plan_id} ---")
    try:
        freshwater_plan = FreshwaterPlan.objects.get(pk=freshwater_plan_id)
        
        # --- Part 1: Generate Plan Text using RAG ---
        _generate_plan_text(freshwater_plan)

        # --- Part 1.5: Generate Static Map Image ---
        _generate_static_map_image(freshwater_plan)

        # --- Part 2: Generate Watermarked PDF Preview ---
        _generate_pdf_preview(freshwater_plan)

        logger.info(f"--- Plan generation and PDF preview complete for ID: {freshwater_plan_id} ---")

    except FreshwaterPlan.DoesNotExist:
        logger.error(f"Task failed: FreshwaterPlan with ID {freshwater_plan_id} does not exist.")
        # Do not retry if the object doesn't exist.
    except Exception as e:
        logger.error(f"An error occurred during plan generation for ID {freshwater_plan_id}: {e}", exc_info=True)
        # Retry the task on failure
        raise self.retry(exc=e)

def _generate_plan_text(freshwater_plan: FreshwaterPlan):
    """Generates the text content of the plan and saves it to the model."""
    # If content already exists, skip regeneration to make the task idempotent.
    if freshwater_plan.generated_plan:
        logger.info(f"Plan text for ID {freshwater_plan.pk} already exists. Skipping generation.")
        return

    logger.info(f"[1/3] Initializing RAG components for plan ID: {freshwater_plan.pk}")
    # Use the singleton service to ensure model consistency across the app
    embeddings = get_embedding_model()
    vector_store_path = os.path.join(settings.BASE_DIR, 'vector_store')
    vector_store = Chroma(persist_directory=vector_store_path, embedding_function=embeddings)
    
    # NOTE: The retriever can be enhanced with a metadata filter if your vector store
    # documents have a 'council' field in their metadata.
    # retriever = vector_store.as_retriever(search_kwargs={'k': 5, 'filter': {'council': freshwater_plan.council}})
    retriever = vector_store.as_retriever(search_kwargs={'k': 5})

    # The model 'llama3-8b-8192' has been decommissioned by Groq.
    # We are updating to a current, recommended model.
    # See: https://console.groq.com/docs/models
    llm = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=settings.GROQ_API_KEY)

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
    # Enhance the question with the dynamic data fetched in the previous step.
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
    # Use update_fields for efficiency and to avoid potential race conditions.
    freshwater_plan.save(update_fields=['generated_plan', 'updated_at'])
    logger.info(f"Plan text generation complete for ID: {freshwater_plan.pk}")

def _generate_static_map_image(freshwater_plan: FreshwaterPlan):
    """
    Fetches a static map image from the LINZ Data Service (LDS) WMS
    and saves it to the FreshwaterPlan's map_image field.
    This uses the LINZ Basemaps API, which is simpler than WMS.
    """
    if freshwater_plan.map_image:
        logger.info(f"Map image for ID {freshwater_plan.pk} already exists. Skipping generation.")
        return
    logger.info(f"Generating static map image for plan ID: {freshwater_plan.pk}")

    # 1. Transform coordinates to NZTM (EPSG:2193) as required by many LINZ services for BBOX.
    try:
        # Use the utility function to transform coordinates.
        lon_nztm, lat_nztm = transform_coords(freshwater_plan.longitude, freshwater_plan.latitude, 4326, 2193)
    except Exception as e:
        logger.error(f"Coordinate transformation failed for plan {freshwater_plan.pk}: {e}")
        # If transformation fails, we cannot proceed.
        return

    # 2. Define a 1km x 1km bounding box around the center point.
    half_size = 500  # meters
    bbox_nztm = (
        lon_nztm - half_size,
        lat_nztm - half_size,
        lon_nztm + half_size,
        lat_nztm + half_size,
    )

    # 3. Construct the LINZ Basemaps GetMap request.
    # This is a simple RESTful API, not a full WMS.
    # See: https://www.linz.govt.nz/data/linz-data-service/guides-and-documentation/wms-and-wmts-guide
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
        response.raise_for_status()  # Raise an exception for bad status codes

        file_name = f"map_plan_{freshwater_plan.pk}.png"
        # Use save=False and a separate model save for consistency
        freshwater_plan.map_image.save(file_name, ContentFile(response.content), save=False)
        freshwater_plan.save(update_fields=['map_image', 'updated_at'])
        logger.info(f"Saved static map image for plan ID: {freshwater_plan.pk}")
    except requests.exceptions.RequestException as e:
        logger.warning(f"Could not fetch static map for plan ID {freshwater_plan.pk}: {e}")

def _generate_pdf_preview(freshwater_plan: FreshwaterPlan):
    """
    Renders the plan to HTML, converts to PDF, and stamps a watermark.
    """
    # If PDF already exists, skip regeneration to make the task idempotent.
    if freshwater_plan.pdf_preview:
        logger.info(f"PDF preview for ID {freshwater_plan.pk} already exists. Skipping generation.")
        return

    if not PDF_DEPS_AVAILABLE:
        logger.warning("PDF generation libraries not installed; skipping PDF preview generation.")
        return

    logger.info(f"Generating PDF preview for plan ID: {freshwater_plan.pk}")
    
    # 1. Render the plan content to an HTML string.
    # We'll need a dedicated template for the PDF.
    html_string = render_to_string('doc_generator/pdf_template.html', {'plan': freshwater_plan})

    # 2. Convert the HTML to a PDF in memory using xhtml2pdf.
    pdf_stream = io.BytesIO()
    pisa_status = pisa.CreatePDF(
        io.StringIO(html_string),  # The HTML source
        dest=pdf_stream             # The PDF destination
    )
    if pisa_status.err:
        raise Exception(f"PDF generation failed with xhtml2pdf: {pisa_status.err}")
    pdf_bytes = pdf_stream.getvalue()

    # 3. Create the watermark and stamp it onto the PDF.
    watermark = _create_watermark_pdf()
    final_pdf_stream = _stamp_pdf(pdf_bytes, watermark)

    # 4. Save the final watermarked PDF to the model's FileField.
    # With the 'prefork' worker, we no longer need the eventlet tpool workaround
    # and can use the standard Django save method directly.
    file_name = f"preview_plan_{freshwater_plan.pk}.pdf"
    file_content = final_pdf_stream.read()
    # The first save attaches the file to the model field in memory.
    freshwater_plan.pdf_preview.save(file_name, ContentFile(file_content), save=False)
    # The second save persists only the changed fields to the database.
    freshwater_plan.save(update_fields=['pdf_preview', 'updated_at'])

    logger.info(f"Saved watermarked PDF preview for plan ID: {freshwater_plan.pk}")

def _get_address_from_coords(lat: float, lon: float) -> str:
    """
    Performs a reverse geocoding lookup to get an address from coordinates.
    Uses the free Nominatim service from OpenStreetMap.
    """
    # Nominatim requires a specific User-Agent header for its usage policy.
    # See: https://operations.osmfoundation.org/policies/nominatim/
    headers = {
        'User-Agent': 'WadeAutomation/1.0 (contact@example.com)' # Replace with a real contact
    }
    url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}"
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        address = data.get('display_name')
        if not address:
            logger.warning(f"Reverse geocoding for lat={lat}, lon={lon} returned no address.")
            return "Address not found."
        return address
    except requests.RequestException as e:
        logger.warning(f"Reverse geocoding request failed for lat={lat}, lon={lon}: {e}")
        return "Could not retrieve address due to a network error."
    except Exception as e:
        logger.error(f"An unexpected error occurred during reverse geocoding: {e}", exc_info=True)
        return "An error occurred while retrieving the address."

def _get_parcel_features_from_wfs(lat: float, lon: float) -> list | None:
    """
    Queries the LINZ WFS for the NZ Primary Parcels layer and returns the
    full feature list for the parcel(s) at the given coordinates.
    """
    wfs_url = f"https://data.linz.govt.nz/services;key={settings.LINZ_API_KEY}/wfs"
    params = {
        'service': 'WFS',
        'version': '2.0.0',
        'request': 'GetFeature',
        'typeNames': 'layer-50772',  # NZ Primary Parcels
        'outputFormat': 'application/json',
        'srsName': 'urn:ogc:def:crs:EPSG::4326', # Ensure output is in WGS84
        'cql_filter': f"INTERSECTS(shape, SRID=4326;POINT({lon} {lat}))",
        'count': 5 # Limit to 5 features
    }
    try:
        response = requests.get(wfs_url, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
        if data and data.get('features'):
            logger.info(f"WFS fallback query successful for ({lat}, {lon}). Found {len(data['features'])} features.")
            # Return the list of features directly
            return data['features']
        else:
            logger.warning(f"WFS fallback query for lat={lat}, lon={lon} returned no features.")
            return None
    except requests.RequestException as e:
        logger.error(f"WFS fallback request for parcel details failed: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during WFS fallback query: {e}", exc_info=True)
        return None

def _update_progress(plan_id: int, message: str, status: str = "pending"):
    """
    Helper function to append a progress update to the plan's log.
    This ensures each update is saved to the database immediately, making it
    visible to the polling front-end.
    """
    try:
        plan = FreshwaterPlan.objects.get(pk=plan_id)
        # Ensure generation_progress is a list
        if not isinstance(plan.generation_progress, list):
            plan.generation_progress = []
        
        plan.generation_progress.append({"message": message, "status": status})
        plan.save(update_fields=['generation_progress', 'updated_at'])
    except FreshwaterPlan.DoesNotExist:
        logger.warning(f"_update_progress called for non-existent plan ID {plan_id}")

@shared_task(bind=True, autoretry_for=(requests.exceptions.HTTPError, Exception,), retry_kwargs={'max_retries': 2, 'retry_backoff': True}, soft_time_limit=300, time_limit=360)
def populate_admin_details_task(self, freshwater_plan_id):
    """
    Populates administrative details and provides live progress updates.
    1. Identifies council and logs progress.
    2. Runs a teaser RAG query for an initial insight.
    3. Fetches address and legal titles concurrently.
    4. Fetches catchment data from ArcGIS, enriches the RAG store, and updates the plan.
    5. Sets status to READY.
    """
    try:
        plan = FreshwaterPlan.objects.get(pk=freshwater_plan_id)
        
        # Step 1: Log Council Confirmation
        _update_progress(plan.pk, f"Verifying council authority for {plan.council} area...", "pending")
        try:
            # Use the singleton service to ensure model consistency
            embeddings = get_embedding_model()
            vector_store_path = os.path.join(settings.BASE_DIR, 'vector_store')
            vector_store = Chroma(persist_directory=vector_store_path, embedding_function=embeddings)
            retriever = vector_store.as_retriever(search_kwargs={'k': 1})
            llm = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=settings.GROQ_API_KEY)

            template = """Based on the context for the '{council}' area, what is the official name of the regional council or unitary authority? Provide only the name.
            Context: {context}
            Answer:"""
            prompt = PromptTemplate.from_template(template)
            rag_chain = ({"context": retriever, "council": lambda x: plan.council} | prompt | llm | StrOutputParser())
            plan.council_authority_name = rag_chain.invoke(f"Official name for {plan.council}")
        except Exception as e:
            logger.warning(f"RAG query for council authority name failed for plan {plan.pk}: {e}")
            plan.council_authority_name = f"Could not verify '{plan.council}'"
        _update_progress(plan.pk, f"Location confirmed in the {plan.council} area.", "complete")
        _update_progress(plan.pk, "The full plan will involve a risk analysis based on catchment data, followed by a detailed assessment and mitigation plan.", "info")

        # Step 2: Run Teaser RAG Query
        _update_progress(plan.pk, "Running preliminary analysis of regional context...", "pending")
        try:
            retriever.search_kwargs={'k': 2}
            template = """Based on the following context for the '{council}' area, what is the single most important environmental risk or regulation a farmer should be aware of? Be very brief and state it in one sentence.
            Context: {context}
            Answer:"""
            prompt = PromptTemplate.from_template(template)

            rag_chain = ({"context": retriever, "council": lambda x: plan.council} | prompt | llm | StrOutputParser())
            teaser_insight = rag_chain.invoke(f"Key environmental consideration for {plan.council}")
            _update_progress(plan.pk, f"Initial analysis: {teaser_insight}", "complete")
        except Exception as e:
            logger.warning(f"Teaser RAG query failed for plan {plan.pk}: {e}")
            _update_progress(plan.pk, "Could not perform initial analysis.", "warning")

        # Step 3: Fetch Admin Details
        _update_progress(plan.pk, "Fetching address and property title information...", "pending")
        # Using gevent.spawn for concurrent non-blocking I/O
        address_thread = gevent.spawn(_get_address_from_coords, plan.latitude, plan.longitude)

        plan.farm_address = address_thread.get()
        _update_progress(plan.pk, "Address retrieved.", "complete")

        # Step 4: Fetch Catchment Name, Soil, and Slope data
        _update_progress(plan.pk, "Fetching regional environmental data...", "pending")
        try:
            # These URLs should be verified and ideally stored in settings
            parcel_layer_id = 53682 # Koordinates "NZ Land Parcels"
            # Corrected domain for ArcGIS services
            soil_url = 'https://services3.arcgis.com/v5RzLI7nHYeFImL4/arcgis/rest/services/Freshwater_farm_plan_contextual_data_hosted/FeatureServer/5/query'
            # New authoritative URL for Southland FMUs from ArcGIS
            fmu_url = 'https://services3.arcgis.com/v5RzLI7nHYeFImL4/arcgis/rest/services/Freshwater_farm_plan_contextual_data_hosted/FeatureServer/1/query'

            # Fetch Catchment/FMU Name from ArcGIS
            catchment_data = _query_arcgis_vector(fmu_url, plan.longitude, plan.latitude)
            if isinstance(catchment_data, list) and catchment_data:
                # The correct property for this ArcGIS layer is 'Zone'.
                # The attributes are now nested in the feature object.
                raw_catchment_name = catchment_data[0].get('properties', {}).get('Zone', 'Not found')
                # Normalize the name by removing "catchment" and trimming whitespace.
                normalized_name = raw_catchment_name.lower().replace('catchment', '').strip().title()
                plan.catchment_name = normalized_name
            else:
                plan.catchment_name = "Catchment/FMU not found at this location via ArcGIS."

            # Fetch Parcel data from Koordinates
            parcel_data = _query_koordinates_vector(parcel_layer_id, plan.longitude, plan.latitude, settings.KOORDINATES_API_KEY, radius=50)
            if isinstance(parcel_data, list) and parcel_data:
                logger.info(f"Koordinates layer {parcel_layer_id} query successful.")
                plan.legal_land_titles = ", ".join([f.get('properties', {}).get('appellation', '') for f in parcel_data])
                plan.total_farm_area_ha = sum(f.get('properties', {}).get('area_ha', 0) for f in parcel_data if f.get('properties', {}).get('area_ha'))
            else:
                logger.warning(f"Koordinates layer {parcel_layer_id} returned no data. Falling back to LINZ WFS.")
                wfs_parcel_data = _get_parcel_features_from_wfs(plan.latitude, plan.longitude)
                if wfs_parcel_data:
                    plan.legal_land_titles = ", ".join([f.get('properties', {}).get('appellation', '') for f in wfs_parcel_data])
                    # Use survey_area_ha as it's more consistently available from WFS
                    plan.total_farm_area_ha = sum(f.get('properties', {}).get('survey_area_ha', 0) for f in wfs_parcel_data if f.get('properties', {}).get('survey_area_ha'))
                else:
                    plan.legal_land_titles = "No parcel information found."
                    plan.total_farm_area_ha = None

            # Fetch Soil Type from ArcGIS Vector
            soil_data = _query_arcgis_vector(soil_url, plan.longitude, plan.latitude) # This is for vector properties
            if isinstance(soil_data, list) and soil_data:
                # The attributes are now nested in the feature object.
                attributes = soil_data[0].get('properties', {})
                plan.soil_type = attributes.get('SoilType', 'Unknown Soil')
                
                # Safely handle 'nil' or other non-numeric values for SlopeAngle
                slope_angle_val = attributes.get('SlopeAngle')
                try:
                    plan.arcgis_slope_angle = float(slope_angle_val)
                except (ValueError, TypeError):
                    logger.warning(f"Could not parse ArcGIS SlopeAngle '{slope_angle_val}'. Setting to None.")
                    plan.arcgis_slope_angle = None

                plan.nutrient_leaching_vulnerability = attributes.get('NutrientLeachingVulnerability')
                plan.erodibility = attributes.get('Erodibility')
            else:
                plan.soil_type = "Not available or error during ArcGIS soil query."

            # Fetch Soil Drainage Class
            drainage_service = SoilDrainageService(lat=plan.latitude, lon=plan.longitude)
            plan.soil_drainage_class = drainage_service.get_soil_drainage_class()

            # Always calculate slope from the DEM as the primary method.
            logger.info(f"Calculating slope from DEM for plan {plan.pk}.")
            slope_degrees = _calculate_slope_from_dem(plan.longitude, plan.latitude, settings.KOORDINATES_API_KEY)
            
            if slope_degrees is not None:
                if slope_degrees < 3: slope_class = 'Flat (0-3°)'
                elif slope_degrees < 7: slope_class = 'Gently Undulating (3-7°)'
                elif slope_degrees < 15: slope_class = 'Rolling (7-15°)'
                else: slope_class = 'Steep (>15°)'
                plan.slope_class = f"{slope_class} ({slope_degrees:.1f}°)"
                # Also save the raw angle if the ArcGIS one wasn't available
                if plan.arcgis_slope_angle is None:
                    plan.arcgis_slope_angle = round(slope_degrees, 1)
            else:
                plan.slope_class = "Not available or error during slope query."

            _update_progress(plan.pk, "Regional environmental data retrieved.", "complete")
        except Exception as e:
            logger.warning(f"Failed to fetch regional data for plan {plan.pk}: {e}", exc_info=True)
            _update_progress(plan.pk, "Could not retrieve some regional data.", "warning")

        # Step 5: Finalize
        plan.generation_status = FreshwaterPlan.GenerationStatus.READY
        plan.save(update_fields=[
            'council_authority_name', 'farm_address', 'legal_land_titles', 'total_farm_area_ha', 'catchment_name',
            'soil_type', 'soil_drainage_class', 'slope_class', 'arcgis_slope_angle',
            'nutrient_leaching_vulnerability', 'erodibility',
            'generation_status', 'generation_progress', 'updated_at'
        ])
        logger.info(f"populate_admin_details_task completed for plan ID: {freshwater_plan_id}")

    except FreshwaterPlan.DoesNotExist:
        logger.warning(f"populate_admin_details_task: plan ID {freshwater_plan_id} does not exist")
    except Exception as e:
        logger.error(f"populate_admin_details_task failed for ID {freshwater_plan_id}: {e}", exc_info=True)
        # Use the helper to log the failure, then update the main status
        _update_progress(plan.pk, f"A critical error occurred: {e}", "error")
        plan.generation_status = FreshwaterPlan.GenerationStatus.FAILED
        plan.save(update_fields=['generation_status', 'updated_at'])
        raise