"""
PDF Processor Module for DocGenius.
Handles multi-PDF ingestion, robust page-by-page extraction, metadata preservation,
validation, text cleaning, text density detection, and seamless Gemini Multimodal OCR
for scanned and image-based pages.
"""

import io
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Callable
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
from core.ocr_engine import ocr_page_if_needed, is_page_text_poor


@dataclass
class PageContent:
    """Represents text extracted from a single page with metadata and OCR provenance."""
    doc_name: str
    page_number: int  # 1-indexed
    text: str
    char_count: int
    word_count: int
    is_empty: bool = False
    is_low_density: bool = False
    is_ocr_applied: bool = False


@dataclass
class ProcessedDocument:
    """Represents a fully processed document with its pages, summary stats, and OCR metrics."""
    doc_name: str
    pages: List[PageContent] = field(default_factory=list)
    total_pages: int = 0
    non_empty_pages: int = 0
    total_chars: int = 0
    total_words: int = 0
    estimated_read_time_mins: float = 0.0
    ocr_pages_count: int = 0
    is_encrypted: bool = False
    is_scanned_or_empty: bool = False
    has_low_density_pages: bool = False
    error_message: Optional[str] = None
    extraction_notes: List[str] = field(default_factory=list)

    def get_full_text(self) -> str:
        """Returns the concatenated non-empty text of all pages."""
        return "\n\n".join(p.text for p in self.pages if not p.is_empty)


def clean_extracted_text(text: str) -> str:
    """
    Cleans raw PDF text extraction artifacts:
    - Normalizes line breaks and whitespace
    - Re-joins words broken across lines by hyphens (e.g. 'envi-\nronment' -> 'environment')
    - Preserves structural paragraph breaks
    """
    if not text:
        return ""

    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"(\b[A-Za-z]+)-\n([a-z]+\b)", r"\1\2", t)

    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in t.split("\n")]

    cleaned_paras: List[str] = []
    current_para: List[str] = []

    for line in lines:
        if not line:
            if current_para:
                cleaned_paras.append(" ".join(current_para))
                current_para = []
        else:
            current_para.append(line)

    if current_para:
        cleaned_paras.append(" ".join(current_para))

    return "\n\n".join(cleaned_paras).strip()


def process_pdf(
    file_source: Union[io.BytesIO, bytes, str, Any],
    doc_name: Optional[str] = None,
    enable_ocr: bool = True,
    api_key: Optional[str] = None,
    ocr_progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> ProcessedDocument:
    """
    Extracts text page-by-page from a PDF file stream, path, or UploadedFile.
    Preserves document name and page number metadata.
    Automatically applies Gemini Multimodal OCR to scanned or text-poor pages.
    """
    if doc_name is None:
        if hasattr(file_source, "name"):
            doc_name = getattr(file_source, "name")
        else:
            doc_name = "document.pdf"

    doc = ProcessedDocument(doc_name=doc_name)

    try:
        # Cache raw bytes for rendering/OCR if needed
        raw_bytes = None
        if isinstance(file_source, bytes):
            raw_bytes = file_source
            stream = io.BytesIO(file_source)
        elif isinstance(file_source, str):
            with open(file_source, "rb") as f:
                raw_bytes = f.read()
            stream = io.BytesIO(raw_bytes)
        elif hasattr(file_source, "getvalue"):
            raw_bytes = file_source.getvalue()
            stream = io.BytesIO(raw_bytes)
        elif hasattr(file_source, "read"):
            if hasattr(file_source, "seek"):
                file_source.seek(0)
            raw_bytes = file_source.read()
            if hasattr(file_source, "seek"):
                file_source.seek(0)
            stream = io.BytesIO(raw_bytes)
        else:
            stream = file_source

        reader = PdfReader(stream)

        # Check encryption
        if reader.is_encrypted:
            try:
                decrypt_res = reader.decrypt("")
                if decrypt_res == 0:
                    doc.is_encrypted = True
                    doc.error_message = f"'{doc_name}' is password-protected and cannot be read."
                    return doc
            except Exception as enc_err:
                doc.is_encrypted = True
                doc.error_message = f"'{doc_name}' is encrypted: {str(enc_err)}"
                return doc

        doc.total_pages = len(reader.pages)
        if doc.total_pages == 0:
            doc.error_message = f"'{doc_name}' contains no pages."
            return doc

        total_extracted_chars = 0
        total_extracted_words = 0
        non_empty_count = 0
        ocr_count = 0
        low_density_pages: List[int] = []

        for idx, page in enumerate(reader.pages):
            page_num = idx + 1
            try:
                raw_text = page.extract_text() or ""
            except Exception:
                raw_text = ""

            clean_text = clean_extracted_text(raw_text)
            is_ocr_applied = False

            # Check if this page is text-poor (e.g. image-based scan or isolated heading)
            if enable_ocr and is_page_text_poor(clean_text, min_chars=60):
                if ocr_progress_callback:
                    ocr_progress_callback(page_num, doc.total_pages, f"Applying OCR to Page {page_num}...")

                ocr_result, applied = ocr_page_if_needed(
                    raw_text=clean_text,
                    file_source=raw_bytes or stream,
                    page_num=page_num,
                    api_key=api_key,
                    page_obj_fallback=page,
                    min_chars=60
                )
                if applied and ocr_result:
                    clean_text = clean_extracted_text(ocr_result)
                    is_ocr_applied = True
                    ocr_count += 1

            char_len = len(clean_text)
            words = len(clean_text.split()) if clean_text else 0

            is_empty = (char_len == 0)
            is_low = (0 < char_len < 60) and not is_ocr_applied

            if is_low:
                low_density_pages.append(page_num)

            if not is_empty:
                non_empty_count += 1
                total_extracted_chars += char_len
                total_extracted_words += words

            page_obj = PageContent(
                doc_name=doc_name,
                page_number=page_num,
                text=clean_text,
                char_count=char_len,
                word_count=words,
                is_empty=is_empty,
                is_low_density=is_low,
                is_ocr_applied=is_ocr_applied
            )
            doc.pages.append(page_obj)

        doc.non_empty_pages = non_empty_count
        doc.total_chars = total_extracted_chars
        doc.total_words = total_extracted_words
        doc.ocr_pages_count = ocr_count
        doc.estimated_read_time_mins = round(total_extracted_words / 200.0, 1)

        if ocr_count > 0:
            doc.extraction_notes.append(
                f"Recovered text from {ocr_count} scanned/image page(s) using Gemini Multimodal OCR."
            )

        if low_density_pages:
            doc.has_low_density_pages = True
            doc.extraction_notes.append(
                f"Pages {low_density_pages} contain short text (<60 chars)."
            )

        if total_extracted_chars == 0:
            doc.is_scanned_or_empty = True
            doc.error_message = (
                f"'{doc_name}' appears to be empty or an image-only / scanned PDF. "
                "No selectable text could be extracted."
            )

    except PdfReadError as pre:
        doc.error_message = f"Failed to parse PDF '{doc_name}': Invalid or corrupted PDF ({str(pre)})."
    except Exception as e:
        doc.error_message = f"Unexpected error while processing '{doc_name}': {str(e)}"

    return doc


def process_multiple_pdfs(
    file_sources: List[Any],
    doc_names: Optional[List[str]] = None,
    enable_ocr: bool = True,
    api_key: Optional[str] = None,
    ocr_progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> List[ProcessedDocument]:
    """
    Processes a list of PDF files, returning a list of ProcessedDocument objects.
    Ensures that a failure in one document does not crash the entire pipeline.
    """
    results: List[ProcessedDocument] = []
    for i, file_obj in enumerate(file_sources):
        name = doc_names[i] if (doc_names and i < len(doc_names)) else None
        res = process_pdf(
            file_source=file_obj,
            doc_name=name,
            enable_ocr=enable_ocr,
            api_key=api_key,
            ocr_progress_callback=ocr_progress_callback
        )
        results.append(res)
    return results
