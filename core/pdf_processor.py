"""
PDF Processor Module for DocGenius.
Handles multi-PDF ingestion, robust page-by-page extraction, metadata preservation,
validation, empty page handling, text cleaning, and document statistics.
"""

import io
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError


@dataclass
class PageContent:
    """Represents text extracted from a single page with metadata."""
    doc_name: str
    page_number: int  # 1-indexed
    text: str
    char_count: int
    word_count: int
    is_empty: bool = False
    is_low_density: bool = False  # True if page has very sparse text (< 60 chars)


@dataclass
class ProcessedDocument:
    """Represents a fully processed document with its pages and summary stats."""
    doc_name: str
    pages: List[PageContent] = field(default_factory=list)
    total_pages: int = 0
    non_empty_pages: int = 0
    total_chars: int = 0
    total_words: int = 0
    estimated_read_time_mins: float = 0.0
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

    # Replace windows line endings
    t = text.replace("\r\n", "\n").replace("\r", "\n")

    # Fix hyphenation at line breaks (e.g. 'deter-\nministic' -> 'deterministic')
    t = re.sub(r"(\b[A-Za-z]+)-\n([a-z]+\b)", r"\1\2", t)

    # Normalize excessive spaces on the same line
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in t.split("\n")]

    # Rejoin lines while preserving double newlines (paragraphs)
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
    doc_name: Optional[str] = None
) -> ProcessedDocument:
    """
    Extracts text page-by-page from a PDF file stream, path, or UploadedFile.
    Preserves document name and page number metadata, handling errors gracefully.
    """
    if doc_name is None:
        if hasattr(file_source, "name"):
            doc_name = getattr(file_source, "name")
        else:
            doc_name = "document.pdf"

    doc = ProcessedDocument(doc_name=doc_name)

    try:
        if isinstance(file_source, bytes):
            stream = io.BytesIO(file_source)
        elif isinstance(file_source, str):
            stream = open(file_source, "rb")
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
        low_density_pages: List[int] = []

        for idx, page in enumerate(reader.pages):
            page_num = idx + 1
            try:
                raw_text = page.extract_text() or ""
            except Exception:
                raw_text = ""

            clean_text = clean_extracted_text(raw_text)
            char_len = len(clean_text)
            words = len(clean_text.split()) if clean_text else 0

            is_empty = (char_len == 0)
            is_low = (0 < char_len < 60)

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
                is_low_density=is_low
            )
            doc.pages.append(page_obj)

        doc.non_empty_pages = non_empty_count
        doc.total_chars = total_extracted_chars
        doc.total_words = total_extracted_words
        doc.estimated_read_time_mins = round(total_extracted_words / 200.0, 1)

        if low_density_pages:
            doc.has_low_density_pages = True
            doc.extraction_notes.append(
                f"Pages {low_density_pages} contain very short text snippets (<60 characters). "
                "These pages may be headings, title slides, or scanned notes with embedded images."
            )

        if total_extracted_chars == 0:
            doc.is_scanned_or_empty = True
            doc.error_message = (
                f"'{doc_name}' appears to be empty or an image-only / scanned PDF. "
                "No selectable text could be extracted."
            )
        elif non_empty_count < (doc.total_pages * 0.3) and doc.total_pages > 2:
            doc.extraction_notes.append(
                f"Only {non_empty_count} of {doc.total_pages} pages contain selectable text. "
                "The remaining pages may be scanned images."
            )

    except PdfReadError as pre:
        doc.error_message = f"Failed to parse PDF '{doc_name}': Invalid or corrupted PDF ({str(pre)})."
    except Exception as e:
        doc.error_message = f"Unexpected error while processing '{doc_name}': {str(e)}"

    return doc


def process_multiple_pdfs(
    file_sources: List[Any],
    doc_names: Optional[List[str]] = None
) -> List[ProcessedDocument]:
    """
    Processes a list of PDF files, returning a list of ProcessedDocument objects.
    Ensures that a failure in one document does not crash the entire pipeline.
    """
    results: List[ProcessedDocument] = []
    for i, file_obj in enumerate(file_sources):
        name = doc_names[i] if (doc_names and i < len(doc_names)) else None
        res = process_pdf(file_obj, doc_name=name)
        results.append(res)
    return results
