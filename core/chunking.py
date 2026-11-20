"""
Chunking Module for DocGenius.
Provides intelligent boundary-aware text splitting, contextual heading fusion,
cross-page continuity, configurable chunk size and overlap, and deduplication.
"""

import hashlib
import re
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional
from core.pdf_processor import ProcessedDocument, PageContent


@dataclass
class DocumentChunk:
    """Represents a text chunk with complete source provenance and metadata."""
    chunk_id: str
    doc_name: str
    page_number: int
    text: str
    char_count: int
    word_count: int
    chunk_index: int
    spanned_pages: List[int] = field(default_factory=list)
    has_heading_context: bool = False


def is_heading_or_fragment(text: str) -> bool:
    """
    Detects if a text snippet looks like a heading, title, or isolated fragment
    rather than a complete explanatory paragraph.
    """
    clean = text.strip()
    if not clean:
        return True

    # Very short snippet
    if len(clean) < 80:
        return True

    # Heading patterns like '(Types of Environment)', '1.2 Overview:', 'Task Environment:-'
    if re.match(r"^(\(?[A-Z0-9][^\.\n]{2,60}\)?[:-]?)$", clean):
        return True

    # Ends with colon or dash indicating a list or section lead-in
    if clean.endswith(":") or clean.endswith(":-") or clean.endswith("—"):
        return True

    # Few words with no ending punctuation
    words = clean.split()
    if len(words) <= 8 and not any(clean.endswith(p) for p in [".", "!", "?", ";"]):
        return True

    return False


def _split_text_with_separators(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    separators: Optional[List[str]] = None
) -> List[str]:
    """
    Recursively splits text using natural boundaries (paragraphs, sentences, words)
    to avoid breaking words or mid-sentence semantics.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""]

    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    chosen_sep = ""
    for sep in separators:
        if sep == "" or sep in text:
            chosen_sep = sep
            break

    splits = text.split(chosen_sep) if chosen_sep != "" else list(text)

    chunks: List[str] = []
    current_chunk: List[str] = []
    current_length = 0

    for s in splits:
        piece = s + chosen_sep if chosen_sep != "" else s
        piece_len = len(piece)

        if current_length + piece_len > chunk_size and current_chunk:
            combined = "".join(current_chunk).strip()
            if combined:
                chunks.append(combined)

            # Build overlap from the end of current_chunk
            overlap_pieces: List[str] = []
            overlap_len = 0
            for prev_p in reversed(current_chunk):
                if overlap_len + len(prev_p) <= chunk_overlap:
                    overlap_pieces.insert(0, prev_p)
                    overlap_len += len(prev_p)
                else:
                    break

            current_chunk = overlap_pieces + [piece]
            current_length = overlap_len + piece_len
        else:
            current_chunk.append(piece)
            current_length += piece_len

    if current_chunk:
        combined = "".join(current_chunk).strip()
        if combined:
            chunks.append(combined)

    return chunks


def chunk_document(
    doc: ProcessedDocument,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    min_chunk_size: int = 100,
    deduplicate: bool = True
) -> List[DocumentChunk]:
    """
    Chunks a ProcessedDocument with contextual heading fusion and metadata retention.
    Prevents headings and short fragments from being stranded without explanatory context.
    """
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size // 4)

    # Step 1: Collect non-empty page segments with page provenance
    page_segments: List[Dict[str, Any]] = []
    for page in doc.pages:
        if page.is_empty or not page.text.strip():
            continue
        page_segments.append({
            "page_number": page.page_number,
            "text": page.text.strip()
        })

    if not page_segments:
        return []

    # Step 2: Contextual Fusion & Boundary-Aware Splitting
    # If a page or block is just a heading/fragment, fuse it with the next block
    fused_blocks: List[Dict[str, Any]] = []
    pending_prefix = ""
    pending_pages: List[int] = []

    for seg in page_segments:
        p_num = seg["page_number"]
        p_text = seg["text"]

        if is_heading_or_fragment(p_text):
            # Accumulate heading/fragment to prefix the next content
            if pending_prefix:
                pending_prefix += "\n\n" + p_text
            else:
                pending_prefix = p_text
            if p_num not in pending_pages:
                pending_pages.append(p_num)
            continue
        else:
            if pending_prefix:
                full_text = f"{pending_prefix}\n\n{p_text}"
                pages = pending_pages + [p_num]
                has_heading = True
                pending_prefix = ""
                pending_pages = []
            else:
                full_text = p_text
                pages = [p_num]
                has_heading = False

            fused_blocks.append({
                "text": full_text,
                "primary_page": pages[0],
                "pages": pages,
                "has_heading": has_heading
            })

    # If document ended with an orphan heading/fragment
    if pending_prefix:
        if fused_blocks:
            # Append orphan heading to the last block
            fused_blocks[-1]["text"] += f"\n\n{pending_prefix}"
            for p in pending_pages:
                if p not in fused_blocks[-1]["pages"]:
                    fused_blocks[-1]["pages"].append(p)
        else:
            # Document only had heading/fragment
            fused_blocks.append({
                "text": pending_prefix,
                "primary_page": pending_pages[0] if pending_pages else 1,
                "pages": pending_pages or [1],
                "has_heading": True
            })

    # Step 3: Split large blocks respecting chunk_size & overlap
    all_chunks: List[DocumentChunk] = []
    seen_hashes: Set[str] = set()
    global_chunk_idx = 0

    for block in fused_blocks:
        raw_text = block["text"]
        primary_page = block["primary_page"]
        spanned_pages = block["pages"]
        has_heading = block["has_heading"]

        raw_splits = _split_text_with_separators(
            raw_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        for s in raw_splits:
            cleaned = s.strip()
            if not cleaned or len(cleaned) < 15:
                continue

            if deduplicate:
                normalized = re.sub(r"\s+", " ", cleaned).lower()
                chunk_hash = hashlib.md5(normalized.encode("utf-8")).hexdigest()
                if chunk_hash in seen_hashes:
                    continue
                seen_hashes.add(chunk_hash)

            chunk_id = f"{doc.doc_name}_p{primary_page}_c{global_chunk_idx}"
            chunk_obj = DocumentChunk(
                chunk_id=chunk_id,
                doc_name=doc.doc_name,
                page_number=primary_page,
                text=cleaned,
                char_count=len(cleaned),
                word_count=len(cleaned.split()),
                chunk_index=global_chunk_idx,
                spanned_pages=spanned_pages,
                has_heading_context=has_heading
            )
            all_chunks.append(chunk_obj)
            global_chunk_idx += 1

    return all_chunks


def chunk_multiple_documents(
    docs: List[ProcessedDocument],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    deduplicate: bool = True
) -> List[DocumentChunk]:
    """
    Chunks a collection of ProcessedDocuments into a single ordered list of DocumentChunks.
    """
    combined_chunks: List[DocumentChunk] = []
    seen_hashes: Set[str] = set()
    global_idx = 0

    for doc in docs:
        doc_chunks = chunk_document(
            doc=doc,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            deduplicate=False
        )
        for c in doc_chunks:
            if deduplicate:
                norm = re.sub(r"\s+", " ", c.text).lower()
                h = hashlib.md5(norm.encode("utf-8")).hexdigest()
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)

            c.chunk_index = global_idx
            combined_chunks.append(c)
            global_idx += 1

    return combined_chunks
