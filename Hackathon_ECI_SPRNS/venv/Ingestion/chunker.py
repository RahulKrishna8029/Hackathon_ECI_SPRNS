from typing import List, Dict, TypedDict, Optional
import os
import numpy as np

# PDF/Text handling
from pathlib import Path
from pdfminer.high_level import extract_text
import pytesseract
from PIL import Image

# NLP utilities
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Ensure punkt tokenizer is available
nltk.download("punkt", quiet=True)


class RawDocument(TypedDict):
    doc_id: str
    path: str
    entity_id: str
    source_type: str
    raw_text: str
    meta: Dict


class Chunk(TypedDict):
    chunk_id: str
    doc_id: str
    entity_id: str
    raw_text: str
    page_num: int
    start_char: int
    end_char: int
    tokens: int
    ocr_confidence: float
    chunk_embed: Optional[List[float]]


# -----------------------
# Ingestion
# -----------------------

def ingest_files(paths: List[str]) -> List[RawDocument]:
    """
    Load raw documents from given paths.
    - PDF: extract text with pdfminer; fallback OCR for images
    - TXT: direct read
    - CSV: flatten rows into text
    Returns list of RawDocument dicts.
    """
    docs: List[RawDocument] = []

    for path in paths:
        ext = Path(path).suffix.lower()
        doc_id = Path(path).stem
        entity_id = "unknown"

        raw_text = ""
        meta: Dict = {}
        source_type = ext.strip(".")

        if ext == ".pdf":
            try:
                raw_text = extract_text(path)
                meta["ocr_confidence"] = 1.0 if raw_text.strip() else 0.0
            except Exception:
                # fallback: OCR each page
                try:
                    img = Image.open(path)
                    raw_text = pytesseract.image_to_string(img)
                    meta["ocr_confidence"] = 0.6  # mark lower confidence
                except Exception as e:
                    raw_text = ""
                    meta["ocr_confidence"] = 0.0
                    meta["error"] = str(e)

        elif ext in [".txt", ".md"]:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                raw_text = f.read()
            meta["ocr_confidence"] = 1.0

        elif ext == ".csv":
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                rows = f.readlines()
            raw_text = "\n".join([row.strip() for row in rows])
            meta["ocr_confidence"] = 1.0

        else:
            meta["ocr_confidence"] = 0.0
            meta["error"] = f"Unsupported file type: {ext}"

        doc: RawDocument = {
            "doc_id": doc_id,
            "path": path,
            "entity_id": entity_id,
            "source_type": source_type,
            "raw_text": raw_text,
            "meta": meta,
        }
        docs.append(doc)

    return docs


# -----------------------
# Chunking
# -----------------------

def _sanitize_tokens(tokens: List[str], max_word_len: int = 40) -> List[str]:
    """Drop or truncate very long tokens (hashes, base64, etc)."""
    clean = []
    for t in tokens:
        if len(t) > max_word_len:
            clean.append(t[:max_word_len] + "…")
        else:
            clean.append(t)
    return clean


def chunk_document(doc: RawDocument, max_tokens: int = 1500, overlap: int = 200) -> List[Chunk]:
    """
    Split doc.raw_text into overlapping chunks with heuristics.
    - Empty doc -> []
    - Sentence-aware splitting
    - Preserve line-based chunks for receipts/tables
    - Handle OCR confidence
    """
    if not doc["raw_text"].strip():
        return []

    text = doc["raw_text"]

    # Heuristic: if many line breaks, treat as table/receipt
    if text.count("\n") > len(text) / 80:
        sentences = text.splitlines()
    else:
        sentences = sent_tokenize(text)

    tokens = []
    offsets = []
    cursor = 0
    for sent in sentences:
        for w in word_tokenize(sent):
            w_clean = _sanitize_tokens([w])[0]
            tokens.append(w_clean)
            start = text.find(w, cursor)
            end = start + len(w)
            offsets.append((start, end))
            cursor = end

    chunks: List[Chunk] = []
    i = 0
    chunk_index = 0
    while i < len(tokens):
        j = min(i + max_tokens, len(tokens))
        chunk_tokens = tokens[i:j]
        if not chunk_tokens:
            break
        start_char = offsets[i][0]
        end_char = offsets[j - 1][1]
        chunk_text = " ".join(chunk_tokens)

        c: Chunk = {
            "chunk_id": f"{doc['doc_id']}_c{chunk_index}",
            "doc_id": doc["doc_id"],
            "entity_id": doc["entity_id"],
            "raw_text": chunk_text,
            "page_num": 0,  # placeholder; could be extracted if available
            "start_char": start_char,
            "end_char": end_char,
            "tokens": len(chunk_tokens),
            "ocr_confidence": doc["meta"].get("ocr_confidence", 1.0),
            "chunk_embed": None,
        }
        chunks.append(c)
        chunk_index += 1
        i = j - overlap if j - overlap > 0 else j

    return chunks


# -----------------------
# Embeddings
# -----------------------

def compute_chunk_embedding(chunk_text: str, embedder) -> np.ndarray:
    """
    Compute seed vector for chunk using provided embedder.
    - embedder must have an `.encode(text)` method (e.g. sentence-transformers).
    """
    if not chunk_text.strip():
        return np.zeros(1)

    vector = embedder.encode(chunk_text)
    if isinstance(vector, list):
        vector = np.array(vector)
    # normalize
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector
