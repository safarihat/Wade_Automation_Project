
import os
import re
import logging
from typing import List, Iterator

from langchain.docstore.document import Document
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Element, Header, Title, Table

logger = logging.getLogger(__name__)

def _extract_keywords_from_text(text: str, max_keywords: int = 5) -> List[str]:
    """Extracts simple keywords from a text string."""
    words = re.findall(r'\b\w+\b', text.lower())
    return list(dict.fromkeys(words))[:max_keywords]

def process_pdf_with_unstructured(file_path: str) -> Iterator[Document]:
    """
    Processes a single PDF using unstructured, creating structured Documents.
    """
    try:
        elements: List[Element] = partition_pdf(
            filename=file_path,
            strategy="hi_res",
            infer_table_structure=True,
            model_name="yolox"
        )
    except Exception as e:
        logger.error(f"Failed to process {file_path} with unstructured: {e}")
        return

    current_section_title = "Unknown"
    document_title = os.path.basename(file_path)

    for element in elements:
        if isinstance(element, (Header, Title)):
            current_section_title = element.text
            continue

        page_content = element.metadata.text_as_html or element.text if isinstance(element, Table) else element.text

        if not page_content.strip():
            continue

        metadata = {
            "document_title": document_title,
            "document_source": os.path.basename(file_path),
            "page_number": element.metadata.page_number,
            "section_title": current_section_title,
            "keywords": _extract_keywords_from_text(current_section_title or page_content)
        }

        yield Document(page_content=page_content, metadata=metadata)
