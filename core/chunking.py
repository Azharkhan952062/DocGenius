"""
Chunking Module for DocGenius.
Provides intelligent boundary-aware text splitting, page-level metadata retention,
configurable chunk size and overlap, and deduplication.
"""

import hashlib
import re
from dataclasses import dataclass
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

    # Find the first separator that appears in text
    chosen_sep = ""
    for sep in separators:
        if sep == "" or sep in text:
            chosen_sep = sep
            break

    splits = text.split(chosen_sep) if chosen_sep != "" else list(text)

    # Recombine splits respecting chunk_size and chunk_overlap
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
    deduplicate: bool = True
) -> List[DocumentChunk]:
    """
    Chunks a ProcessedDocument page by page, preserving page metadata.
    Avoids duplicate chunks and drops whitespace-only slices.
    """
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size // 4)

    all_chunks: List[DocumentChunk] = []
    seen_hashes: Set[str] = set()
    global_chunk_idx = 0

    for page in doc.pages:
        if page.is_empty or not page.text.strip():
            continue

        page_text = page.text
        raw_chunks = _split_text_with_separators(
            page_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        for raw_c in raw_chunks:
            cleaned = raw_c.strip()
            if not cleaned or len(cleaned) < 15:  # Ignore trivial noise
                continue

            if deduplicate:
                # Normalize whitespace for hashing
                normalized = re.sub(r"\s+", " ", cleaned).lower()
                chunk_hash = hashlib.md5(normalized.encode("utf-8")).hexdigest()
                if chunk_hash in seen_hashes:
                    continue
                seen_hashes.add(chunk_hash)

            chunk_id = f"{doc.doc_name}_p{page.page_number}_c{global_chunk_idx}"
            chunk_obj = DocumentChunk(
                chunk_id=chunk_id,
                doc_name=doc.doc_name,
                page_number=page.page_number,
                text=cleaned,
                char_count=len(cleaned),
                word_count=len(cleaned.split()),
                chunk_index=global_chunk_idx
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
            deduplicate=False  # Cross-document deduplication handled below
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
