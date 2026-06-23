"""Custom ZeroEntropy embedder for mem0ai.

mem0ai ships a ``zero_entropy`` *reranker* but no ZeroEntropy *embedder*. This
module registers one (via ``server.register_embedders()``) so that
``MEM0_EMBED_PROVIDER=zeroentropy`` embeds with ZeroEntropy's ``zembed-1`` model
— independently of whatever reranker is configured.

ZeroEntropy embeddings are *asymmetric*: queries and documents are embedded at
slightly different points in space, so we map mem0's ``memory_action`` to the
API's ``input_type`` ("add"/"update" -> ``document``, "search" -> ``query``).

``zembed-1`` is a matryoshka model — the dimensionality is selectable from
{2560, 1280, 640, 320, 160, 80, 40} (default 2560). When ``embedding_dims`` is
set we forward it as ``dimensions`` so the vectors match the width the Qdrant
collection was created with; the two MUST agree.

The ``zeroentropy`` package is imported lazily so it is only required when this
embedder is actually selected (install via the ``[zeroentropy]`` extra).
"""

from __future__ import annotations

import os
from typing import Literal, Optional

from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.embeddings.base import EmbeddingBase

# mem0 memory_action -> ZeroEntropy input_type (asymmetric retrieval).
_INPUT_TYPE = {"add": "document", "update": "document", "search": "query"}


class ZeroEntropyEmbedding(EmbeddingBase):
    """Embedding provider backed by ZeroEntropy's ``zembed-1`` model."""

    def __init__(self, config: Optional[BaseEmbedderConfig] = None):
        super().__init__(config)

        try:
            from zeroentropy import ZeroEntropy
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "The 'zeroentropy' package is required for the ZeroEntropy embedder. "
                "Install it with: pip install 'mem0-mcp-selfhosted[zeroentropy]'."
            ) from exc

        self.config.model = self.config.model or "zembed-1"
        # Only forward `dimensions` when explicitly configured; otherwise let the
        # API apply its model default (2560 for zembed-1).
        self._dimensions = self.config.embedding_dims

        api_key = self.config.api_key or os.getenv("ZERO_ENTROPY_API_KEY")
        if not api_key:
            raise ValueError(
                "ZeroEntropy API key required. Set ZERO_ENTROPY_API_KEY (or "
                "MEM0_EMBED_API_KEY), or pass api_key in the embedder config."
            )
        self.client = ZeroEntropy(api_key=api_key)

    @staticmethod
    def _input_type(memory_action: Optional[str]) -> str:
        return _INPUT_TYPE.get(memory_action or "add", "document")

    def _embed_kwargs(self, value, memory_action: Optional[str]) -> dict:
        kwargs = {
            "input": value,
            "input_type": self._input_type(memory_action),
            "model": self.config.model,
            "encoding_format": "float",
        }
        if self._dimensions:
            kwargs["dimensions"] = self._dimensions
        return kwargs

    def embed(
        self,
        text,
        memory_action: Optional[Literal["add", "search", "update"]] = None,
    ):
        """Embed a single string. Returns the embedding vector (list of floats)."""
        resp = self.client.models.embed(**self._embed_kwargs(text, memory_action))
        return resp.results[0].embedding

    def embed_batch(
        self,
        texts,
        memory_action: Optional[Literal["add", "search", "update"]] = "add",
    ):
        """Embed many strings in a single API call (input order preserved)."""
        resp = self.client.models.embed(**self._embed_kwargs(list(texts), memory_action))
        return [r.embedding for r in resp.results]
