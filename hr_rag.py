import json
import os
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple, Dict, Any

import faiss
import numpy as np
from loguru import logger
from openai import OpenAI


RAG_DIR = Path(__file__).parent / "rag_store"
INDEX_PATH = RAG_DIR / "hr_policies.index"
META_PATH = RAG_DIR / "hr_policies_meta.json"

EMBED_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
TOP_K_DEFAULT = int(os.getenv("RAG_TOP_K", "4"))
MIN_SCORE_DEFAULT = float(os.getenv("RAG_MIN_SCORE", "0.2"))


class HRPolicyRetriever:
    def __init__(
        self,
        index_path: Path = INDEX_PATH,
        meta_path: Path = META_PATH,
        model: str = EMBED_MODEL,
    ) -> None:
        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"RAG index files not found. Expected {index_path} and {meta_path}. "
                f"Run build_hr_index.py first."
            )

        self._client = OpenAI()
        self._model = model

        logger.info(f"Loading FAISS index from {index_path}")
        self._index = faiss.read_index(str(index_path))

        with meta_path.open("r", encoding="utf-8") as f:
            self._chunks: List[Dict[str, Any]] = json.load(f)

        logger.info(
            f"Loaded HR RAG metadata: {len(self._chunks)} chunks, index size={self._index.ntotal}"
        )

    def _embed_query(self, query: str) -> np.ndarray:
        resp = self._client.embeddings.create(model=self._model, input=[query])
        embedding = np.array(resp.data[0].embedding, dtype="float32")
        faiss.normalize_L2(embedding.reshape(1, -1))
        return embedding

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_DEFAULT,
        min_score: float = MIN_SCORE_DEFAULT,
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        if not query.strip():
            return [], []

        emb = self._embed_query(query)
        scores, indices = self._index.search(emb.reshape(1, -1), top_k)
        scores = scores[0].tolist()
        indices = indices[0].tolist()

        results: List[Dict[str, Any]] = []
        result_scores: List[float] = []
        for score, idx in zip(scores, indices):
            if idx == -1:
                continue
            if score < min_score:
                continue
            chunk = self._chunks[idx]
            result = {
                "page": chunk.get("page"),
                "text": chunk.get("text"),
                "score": score,
            }
            results.append(result)
            result_scores.append(score)

        logger.debug(
            f"RAG retrieve for query={query!r}: "
            f"top_k={top_k}, min_score={min_score}, returned={len(results)}"
        )
        return results, result_scores


@lru_cache(maxsize=1)
def get_hr_retriever() -> HRPolicyRetriever:
    return HRPolicyRetriever()


def retrieve_hr_policies_context(
    query: str,
    top_k: int = TOP_K_DEFAULT,
    min_score: float = MIN_SCORE_DEFAULT,
) -> List[str]:
    """Return list of formatted HR policy snippets for use in prompts."""
    retriever = get_hr_retriever()
    results, scores = retriever.retrieve(query, top_k=top_k, min_score=min_score)
    if not results:
        logger.debug("RAG: no HR policy snippets found above score threshold")
        return []

    formatted: List[str] = []
    for r in results:
        page = r.get("page")
        text = r.get("text", "").strip()
        score = r.get("score")
        snippet = f"[HR Policy Page {page}, score={score:.3f}]\n{text}"
        formatted.append(snippet)

    logger.debug(f"RAG: returning {len(formatted)} formatted snippets")
    logger.debug(formatted)
    return formatted


