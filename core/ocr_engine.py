"""
OCR Engine for DocGenius.
Provides intelligent page-by-page optical character recognition for scanned,
photographed, and image-based PDFs using Gemini Multimodal Vision and high-performance
in-memory page rendering via pypdfium2 (with PyPDF2 XObject fallback).
"""

import os
import io
import time
from typing import Optional, List, Union, Any, Tuple
from PIL import Image
import google.generativeai as genai

# Try importing pypdfium2 for high-fidelity vector & scan rendering
try:
    import pypdfium2 as pdfium
    HAS_PYPDFIUM2 = True
except ImportError:
    pdfium = None
    HAS_PYPDFIUM2 = False

OCR_FALLBACK_MODELS = [
    "models/gemini-flash-lite-latest",
    "models/gemini-2.5-flash-lite",
    "models/gemini-3.8-flash",
    "models/gemini-flash-latest"
]

OCR_PROMPT = """You are an expert document transcription system.
Transcribe all handwritten and printed text from this study notes / document page verbatim.
Rules:
1. Preserve all headings, subheadings, bullet points, numbered lists, and formulas accurately.
2. Keep definitions and explanatory paragraphs exactly as written in the notes.
3. Do not omit technical details, diagrams labels, examples, or equations.
4. Output only the clean markdown transcription without conversational preamble.
"""


def is_page_text_poor(text: str, min_chars: int = 60) -> bool:
    """
    Determines whether a page contains insufficient selectable text,
    indicating it is likely an image scan or sparse annotation.
    """
    if not text:
        return True
    cleaned = text.strip()
    return len(cleaned) < min_chars


def render_page_to_image(
    file_source: Union[io.BytesIO, bytes, str, Any],
    page_number_1_indexed: int,
    scale: float = 1.5,
    page_obj_fallback: Optional[Any] = None
) -> Optional[Image.Image]:
    """
    Renders a PDF page to a PIL Image.
    Uses pypdfium2 for rasterization, falling back to PyPDF2 XObject extraction if needed.
    """
    idx_0 = page_number_1_indexed - 1

    # Method 1: pypdfium2 (high fidelity rasterization of scanned/image pages)
    if HAS_PYPDFIUM2:
        try:
            if isinstance(file_source, bytes):
                doc = pdfium.PdfDocument(file_source)
            elif isinstance(file_source, str):
                doc = pdfium.PdfDocument(file_source)
            elif hasattr(file_source, "getvalue"):
                doc = pdfium.PdfDocument(file_source.getvalue())
            elif hasattr(file_source, "read"):
                # Seek to beginning if possible
                if hasattr(file_source, "seek"):
                    file_source.seek(0)
                content = file_source.read()
                if hasattr(file_source, "seek"):
                    file_source.seek(0)
                doc = pdfium.PdfDocument(content)
            else:
                doc = None

            if doc and 0 <= idx_0 < len(doc):
                page = doc[idx_0]
                pil_img = page.render(scale=scale).to_pil()
                return pil_img
        except Exception:
            pass

    # Method 2: Fallback to PyPDF2 XObject image extraction if page_obj_fallback provided
    if page_obj_fallback is not None:
        try:
            p = page_obj_fallback
            if "/Resources" in p:
                res = p["/Resources"]
                if hasattr(res, "get_object"):
                    res = res.get_object()
                if "/XObject" in res:
                    xobj = res["/XObject"]
                    if hasattr(xobj, "get_object"):
                        xobj = xobj.get_object()

                    # Find largest image object by resolution
                    best_img: Optional[Image.Image] = None
                    best_area = 0

                    for k, v in xobj.items():
                        o = v.get_object() if hasattr(v, "get_object") else v
                        if o.get("/Subtype") == "/Image":
                            try:
                                data = o.get_data()
                                im = Image.open(io.BytesIO(data))
                                area = im.size[0] * im.size[1]
                                if area > best_area and area > 20000:  # Ignore tiny icons
                                    best_area = area
                                    best_img = im
                            except Exception:
                                continue
                    if best_img:
                        return best_img
        except Exception:
            pass

    return None


def transcribe_image_with_gemini(
    image: Image.Image,
    api_key: Optional[str] = None,
    preferred_model: str = "models/gemini-flash-lite-latest",
    max_dimension: int = 1200
) -> Optional[str]:
    """
    Transcribes text from a PIL Image using Gemini Multimodal Vision.
    Optimizes the image to max_dimension JPEG to ensure minimal network latency.
    """
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        return None

    genai.configure(api_key=key)

    # Optimize image for fast transmission (<150KB)
    img_copy = image.copy()
    if img_copy.mode in ("RGBA", "P"):
        img_copy = img_copy.convert("RGB")

    img_copy.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img_copy.save(buf, format="JPEG", quality=82)
    buf.seek(0)
    optimized_img = Image.open(buf)

    # Candidate models in order of priority
    candidate_models = [preferred_model]
    for fb in OCR_FALLBACK_MODELS:
        if fb not in candidate_models:
            candidate_models.append(fb)

    last_err = ""
    for m_name in candidate_models:
        try:
            model = genai.GenerativeModel(m_name)
            res = model.generate_content(
                [OCR_PROMPT, optimized_img],
                request_options={"timeout": 25}
            )
            text = res.text.strip()
            if text:
                return text
        except Exception as e:
            last_err = str(e)
            continue

    return None


def ocr_page_if_needed(
    raw_text: str,
    file_source: Any,
    page_num: int,
    api_key: Optional[str] = None,
    page_obj_fallback: Optional[Any] = None,
    min_chars: int = 60
) -> Tuple[str, bool]:
    """
    Evaluates whether a page requires OCR. If text density is poor,
    renders the page and transcribes with Gemini Multimodal Vision.
    Returns (final_text, is_ocr_applied).
    """
    if not is_page_text_poor(raw_text, min_chars=min_chars):
        # Sufficient selectable text exists; bypass OCR to optimize speed
        return raw_text, False

    # Page is text-poor; attempt OCR
    img = render_page_to_image(
        file_source=file_source,
        page_number_1_indexed=page_num,
        page_obj_fallback=page_obj_fallback
    )

    if not img:
        return raw_text, False

    transcription = transcribe_image_with_gemini(img, api_key=api_key)
    if transcription and len(transcription.strip()) > 15:
        # If there was a tiny existing heading (like 22 chars), preserve it at the top
        if raw_text.strip() and raw_text.strip().lower() not in transcription.lower():
            combined = f"{raw_text.strip()}\n\n{transcription.strip()}"
        else:
            combined = transcription.strip()
        return combined, True

    return raw_text, False
