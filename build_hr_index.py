import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple

import faiss
from loguru import logger
from openai import OpenAI
from pypdf import PdfReader


RAG_DIR = Path(__file__).parent / "rag_store"
RAG_DIR.mkdir(exist_ok=True)

INDEX_PATH = RAG_DIR / "hr_policies.index"
META_PATH = RAG_DIR / "hr_policies_meta.json"

PDF_PATH = Path(__file__).parent / "HR-Policies-Manuals.pdf"

EMBED_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def load_pdf_text(pdf_path: Path) -> List[Tuple[int, str]]:
    """Return list of (page_number, page_text)."""
    reader = PdfReader(str(pdf_path))
    pages: List[Tuple[int, str]] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append((i + 1, text))
    logger.info(f"Loaded {len(pages)} pages from {pdf_path}")
    return pages


def chunk_text(text: str, page: int, max_chars: int = 1200, overlap: int = 200) -> List[Dict[str, Any]]:
    """Simple character-based chunking with overlap for robustness."""
    chunks: List[Dict[str, Any]] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                {
                    "page": page,
                    "text": chunk_text,
                    "start": start,
                    "end": end,
                }
            )
        if end == n:
            break
        start = end - overlap
    return chunks


def build_chunks() -> List[Dict[str, Any]]:
    pages = load_pdf_text(PDF_PATH)
    all_chunks: List[Dict[str, Any]] = []
    for page, text in pages:
        page_chunks = chunk_text(text, page)
        all_chunks.extend(page_chunks)
    logger.info(f"Created {len(all_chunks)} chunks from HR policies PDF")
    return all_chunks


def embed_chunks(chunks: List[Dict[str, Any]]) -> Any:
    client = OpenAI()
    texts = [c["text"] for c in chunks]

    logger.info(f"Embedding {len(texts)} chunks with model={EMBED_MODEL}")
    # OpenAI Python SDK v1 embeddings API
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vectors = [e.embedding for e in resp.data]

    import numpy as np

    matrix = np.array(vectors, dtype="float32")
    logger.info(f"Embedding matrix shape: {matrix.shape}")
    return matrix


def build_faiss_index(matrix, chunks: List[Dict[str, Any]]):
    import numpy as np

    dim = matrix.shape[1]
    index = faiss.IndexFlatIP(dim)

    # Normalize for cosine similarity
    faiss.normalize_L2(matrix)

    index.add(matrix)
    logger.info(f"FAISS index built with {index.ntotal} vectors (dim={dim})")

    # Persist index and metadata
    faiss.write_index(index, str(INDEX_PATH))
    with META_PATH.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved index to {INDEX_PATH} and metadata to {META_PATH}")


def main() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"HR policies PDF not found at {PDF_PATH}")

    chunks = build_chunks()
    matrix = embed_chunks(chunks)
    build_faiss_index(matrix, chunks)


if __name__ == "__main__":
    main()


