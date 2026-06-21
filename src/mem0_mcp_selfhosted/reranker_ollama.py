"""Ollama-backed reranker for mem0-mcp-selfhosted.

Uses Ollama's /api/rerank endpoint (available in Ollama v0.5.3+).
Registered with mem0ai's RerankerFactory under the "ollama" provider name.

Supported models (must be pulled first):
  ollama pull bge-reranker-v2-m3
  ollama pull mxbai-rerank-large-v1
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx
from pydantic import Field

from mem0.configs.rerankers.base import BaseRerankerConfig
from mem0.reranker.base import BaseReranker

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "bge-reranker-v2-m3"
_DEFAULT_URL = "http://localhost:11434"
_DEFAULT_TOP_K = 20
_DEFAULT_TIMEOUT = 30.0


class OllamaRerankerConfig(BaseRerankerConfig):
    """Config for the Ollama reranker.

    Inherits provider, model, api_key, top_k from BaseRerankerConfig.
    """

    ollama_base_url: str = Field(default=_DEFAULT_URL)
    timeout: float = Field(default=_DEFAULT_TIMEOUT)


def _extract_text(doc: Dict[str, Any]) -> str:
    """Extract the text field from a mem0ai memory dict."""
    return (
        doc.get("memory")
        or doc.get("text")
        or doc.get("content")
        or str(doc)
    )


class OllamaReranker(BaseReranker):
    """Reranker backed by Ollama's /api/rerank endpoint (v0.5.3+)."""

    def __init__(self, config: OllamaRerankerConfig) -> None:
        self.config = config
        self._base_url = (config.ollama_base_url or _DEFAULT_URL).rstrip("/")
        self._model = config.model or _DEFAULT_MODEL
        self._top_k = config.top_k or _DEFAULT_TOP_K
        self._timeout = config.timeout

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Rerank documents using Ollama's /api/rerank endpoint.

        Args:
            query: The search query.
            documents: List of mem0ai memory dicts, each with a 'memory' field.
            top_k: Maximum results to return (overrides config.top_k).

        Returns:
            Documents sorted by relevance_score descending, with 'rerank_score' added.
            Falls back to original order (score=0.0) on any error.
        """
        if not documents:
            return documents

        effective_top_k = top_k or self._top_k
        doc_texts = [_extract_text(doc) for doc in documents]

        try:
            response = httpx.post(
                f"{self._base_url}/api/rerank",
                json={
                    "model": self._model,
                    "query": query,
                    "documents": doc_texts,
                },
                timeout=self._timeout,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            logger.warning(
                "Ollama rerank failed (%s), returning original order: %s",
                type(exc).__name__,
                exc,
            )
            for doc in documents:
                doc.setdefault("rerank_score", 0.0)
            return documents[:effective_top_k] if effective_top_k else documents

        # Ollama response: {"results": [{"index": 0, "relevance_score": 0.9}, ...]}
        results = data.get("results", [])
        reranked: List[Dict[str, Any]] = []
        for item in results:
            idx = item.get("index")
            score = item.get("relevance_score", 0.0)
            if idx is None or idx >= len(documents):
                continue
            doc = documents[idx].copy()
            doc["rerank_score"] = score
            reranked.append(doc)

        # Sort descending by score (Ollama may already sort, but be explicit)
        reranked.sort(key=lambda d: d.get("rerank_score", 0.0), reverse=True)

        if effective_top_k:
            reranked = reranked[:effective_top_k]

        return reranked
