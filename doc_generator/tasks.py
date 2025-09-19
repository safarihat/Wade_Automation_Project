import os
import io
import logging
import sys
from contextlib import contextmanager
import requests
import eventlet
from celery import shared_task
from django.conf import settings
from django.template.loader import render_to_string
from django.core.files.base import ContentFile
from doc_generator.models import FreshwaterPlan

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
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

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

@shared_task(bind=True, max_retries=3, default_retry_delay=60)
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
        # _generate_static_map_image(freshwater_plan)

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
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    # Explicitly set the device to 'cpu' to avoid "meta tensor" errors
    # that can occur with newer versions of torch and sentence-transformers.
    model_kwargs = {'device': 'cpu'}
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

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
    """
    if freshwater_plan.map_image:
        logger.info(f"Map image for ID {freshwater_plan.pk} already exists. Skipping generation.")
        return

    logger.info(f"Generating static map image for plan ID: {freshwater_plan.pk}")

    # 1. Transform coordinates to NZTM (EPSG:2193) as required by many LINZ services for BBOX.
    center_point_wgs84 = freshwater_plan.location
    center_point_nztm = center_point_wgs84.transform(2193, clone=True)

    # 2. Define a 1km x 1km bounding box around the center point.
    half_size = 500  # meters
    bbox_nztm = (
        center_point_nztm.x - half_size,
        center_point_nztm.y - half_size,
        center_point_nztm.x + half_size,
        center_point_nztm.y + half_size,
    )

    # 3. Construct the LINZ WMS GetMap request using the correct Basemaps service endpoint.
    # The API key is passed as a standard query parameter. We use the data.linz.govt.nz WMS.
    wms_url = "https://data.linz.govt.nz/services/wms/"
    params = {
        'key': settings.LINZ_API_KEY,
        'service': 'WMS',
        'request': 'GetMap',
        'layers': 'layer-50767',  # The layer ID for NZTopo50 on data.linz.govt.nz WMS.
        'styles': '',
        'format': 'image/png',
        'transparent': 'true',
        'version': '1.3.0', # Using a more current WMS version
        'width': 800,
        'height': 800,
        'crs': 'EPSG:2193', # WMS 1.3.0 uses 'crs' instead of 'srs'
        'bbox': ','.join(map(str, bbox_nztm)),
    }

    try:
        # Manually construct the full URL to prevent `requests` from encoding the semicolon in the path.
        query_string = "&".join([f"{k}={v}" for k, v in params.items() if v is not None])
        full_url = f"{wms_url}?{query_string}"
        response = requests.get(full_url, timeout=30) # Pass the fully constructed URL
        response.raise_for_status()  # Raise an exception for bad status codes

        file_name = f"map_plan_{freshwater_plan.pk}.png"
        # Use save=False and a separate model save for consistency
        freshwater_plan.map_image.save(file_name, ContentFile(response.content), save=False)
        freshwater_plan.save(update_fields=['map_image', 'updated_at'])
        logger.info(f"Saved static map image for plan ID: {freshwater_plan.pk}")
    except requests.exceptions.RequestException as e:
        logger.warning(f"Could not fetch static map for plan ID {freshwater_plan.pk}: {e}")

def _generate_pdf_preview(freshwater_plan: FreshwaterPlan):
    """Renders the plan to HTML, converts to PDF, and stamps a watermark."""
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

def _get_parcel_details_from_linz(lat: float, lon: float) -> dict:
    """
    Queries the LINZ WFS for the NZ Primary Parcels layer to get details
    for the parcel at the given coordinates.
    """
    wfs_url = f"https://data.linz.govt.nz/services;key={settings.LINZ_API_KEY}/wfs"
    params = {
        'service': 'WFS',
        'version': '2.0.0',
        'request': 'GetFeature',
        'typeNames': 'layer-50772', # Layer ID for NZ Primary Parcels
        'outputFormat': 'application/json',
        # Use a CQL filter to find the parcel that intersects the point
        'cql_filter': f"INTERSECTS(shape, SRID=4326;POINT({lon} {lat}))"
    }
    try:
        response = requests.get(wfs_url, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
        if data and data.get('features'):
            properties = data['features'][0].get('properties', {})
            # Extract relevant details. 'appellation' is the legal description.
            legal_titles = properties.get('appellation', 'Not found')
            return {'legal_land_titles': legal_titles}
        else:
            logger.warning(f"WFS query for lat={lat}, lon={lon} returned no features.")
            return {'legal_land_titles': 'No parcel information found at this location.'}
    except requests.RequestException as e:
        logger.warning(f"WFS request for parcel details failed: {e}")
        return {'legal_land_titles': 'Could not retrieve parcel information due to a network error.'}
    except Exception as e:
        logger.error(f"An unexpected error occurred during WFS query: {e}", exc_info=True)
        return {'legal_land_titles': 'An error occurred while retrieving parcel information.'}

@shared_task(bind=True, max_retries=0)
def populate_admin_details_task(self, freshwater_plan_id):
    """Populates administrative details for a plan, including reverse geocoding the coordinates to get a farm address."""
    try:
        plan = FreshwaterPlan.objects.get(pk=freshwater_plan_id)
        
        # Spawn green threads to run the network calls concurrently,
        # taking advantage of the eventlet worker pool.
        address_thread = eventlet.spawn(_get_address_from_coords, plan.latitude, plan.longitude)
        parcel_thread = eventlet.spawn(_get_parcel_details_from_linz, plan.latitude, plan.longitude)

        # Wait for both threads to complete and retrieve their results.
        plan.farm_address = address_thread.wait()
        parcel_details = parcel_thread.wait()
        plan.legal_land_titles = parcel_details.get('legal_land_titles')

        plan.generation_status = FreshwaterPlan.GenerationStatus.READY
        plan.save(update_fields=['farm_address', 'legal_land_titles', 'generation_status', 'updated_at'])
        logger.info(f"populate_admin_details_task completed for plan ID: {freshwater_plan_id}")
    except FreshwaterPlan.DoesNotExist:
        logger.warning(f"populate_admin_details_task: plan ID {freshwater_plan_id} does not exist")
    except Exception as e:
        logger.error(f"populate_admin_details_task failed for ID {freshwater_plan_id}: {e}", exc_info=True)
        # Ensure the plan status is updated to FAILED so the UI can react.
        plan.generation_status = FreshwaterPlan.GenerationStatus.FAILED
        plan.save(update_fields=['generation_status', 'updated_at'])
        # Re-raise the exception to mark the Celery task as failed.
        raise