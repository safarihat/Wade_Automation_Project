import os
import io
import logging
from celery import shared_task
from django.conf import settings
from django.template.loader import render_to_string
from django.core.files.base import ContentFile
from .models import FreshwaterPlan

# PDF and Watermarking libraries
try:
    from weasyprint import HTML
    from pypdf import PdfReader, PdfWriter
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
except ImportError as e:
    # This allows the app to load even if these are not installed,
    # but the task will fail if called.
    raise ImportError(
        f"PDF generation libraries not found: {e}. "
        "Please install them with: pip install weasyprint pypdf reportlab"
    ) from e

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
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    vector_store_path = os.path.join(settings.BASE_DIR, 'vector_store')
    vector_store = Chroma(persist_directory=vector_store_path, embedding_function=embeddings)
    
    # NOTE: The retriever can be enhanced with a metadata filter if your vector store
    # documents have a 'council' field in their metadata.
    # retriever = vector_store.as_retriever(search_kwargs={'k': 5, 'filter': {'council': freshwater_plan.council}})
    retriever = vector_store.as_retriever(search_kwargs={'k': 5})

    llm = ChatGroq(model_name="llama3-8b-8192", groq_api_key=settings.GROQ_API_KEY)

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
    question_for_llm = f"Generate a preliminary freshwater farm plan for a property located at latitude {freshwater_plan.latitude}, longitude {freshwater_plan.longitude} within the {freshwater_plan.council} region. Focus on identifying key environmental risks and suggesting initial mitigation strategies based on the provided context."
    generated_content = rag_chain.invoke(question_for_llm)
    logger.info(f"RAG chain invocation complete for plan ID: {freshwater_plan.pk}")

    logger.info(f"[3/3] Saving generated text to database for plan ID: {freshwater_plan.pk}")
    freshwater_plan.generated_plan = generated_content
    # Use update_fields for efficiency and to avoid potential race conditions.
    freshwater_plan.save(update_fields=['generated_plan', 'updated_at'])
    logger.info(f"Plan text generation complete for ID: {freshwater_plan.pk}")

def _generate_pdf_preview(freshwater_plan: FreshwaterPlan):
    """Renders the plan to HTML, converts to PDF, and stamps a watermark."""
    # If PDF already exists, skip regeneration to make the task idempotent.
    if freshwater_plan.pdf_preview:
        logger.info(f"PDF preview for ID {freshwater_plan.pk} already exists. Skipping generation.")
        return

    logger.info(f"Generating PDF preview for plan ID: {freshwater_plan.pk}")
    
    # 1. Render the plan content to an HTML string.
    # We'll need a dedicated template for the PDF.
    html_string = render_to_string('doc_generator/pdf_template.html', {'plan': freshwater_plan})

    # 2. Convert the HTML to a PDF in memory.
    pdf_bytes = HTML(string=html_string).write_pdf()

    # 3. Create the watermark and stamp it onto the PDF.
    watermark = _create_watermark_pdf()
    final_pdf_stream = _stamp_pdf(pdf_bytes, watermark)

    # 4. Save the final watermarked PDF to the model's FileField.
    file_name = f"preview_plan_{freshwater_plan.pk}.pdf"
    freshwater_plan.pdf_preview.save(file_name, ContentFile(final_pdf_stream.read()), save=True)
    logger.info(f"Saved watermarked PDF preview for plan ID: {freshwater_plan.pk}")