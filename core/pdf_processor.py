"""
PDF Processor Module for DocGenius.
Handles multi-PDF ingestion, robust page-by-page extraction, metadata preservation,
validation, empty page handling, and document statistics.
"""

import io
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
    error_message: Optional[str] = None

    def get_full_text(self) -> str:
        """Returns the concatenated non-empty text of all pages."""
        return "\n\n".join(p.text for p in self.pages if not p.is_empty)


def process_pdf(
    file_source: Union[io.BytesIO, bytes, str, Any],
    doc_name: Optional[str] = None
) -> ProcessedDocument:
    """
    Extracts text page-by-page from a PDF file stream, path, or UploadedFile.
    Preserves document name and page number metadata, handling errors gracefully.
    """
    # Determine document name
    if doc_name is None:
        if hasattr(file_source, "name"):
            doc_name = getattr(file_source, "name")
        else:
            doc_name = "document.pdf"

    doc = ProcessedDocument(doc_name=doc_name)

    try:
        # Wrap bytes or path if needed
        if isinstance(file_source, bytes):
            stream = io.BytesIO(file_source)
        elif isinstance(file_source, str):
            stream = open(file_source, "rb")
        else:
            stream = file_source

        # Read PDF
        reader = PdfReader(stream)

        # Check encryption
        if reader.is_encrypted:
            try:
                # Try decrypting with empty password
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

        for idx, page in enumerate(reader.pages):
            page_num = idx + 1
            try:
                raw_text = page.extract_text() or ""
            except Exception as page_err:
                raw_text = ""

            clean_text = raw_text.strip()
            char_len = len(clean_text)
            words = len(clean_text.split()) if clean_text else 0

            is_empty = (char_len == 0)
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
                is_empty=is_empty
            )
            doc.pages.append(page_obj)

        doc.non_empty_pages = non_empty_count
        doc.total_chars = total_extracted_chars
        doc.total_words = total_extracted_words
        doc.estimated_read_time_mins = round(total_extracted_words / 200.0, 1)

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
