"""Unit tests for OllamaReranker and OllamaRerankerConfig.

All tests mock httpx.post — no live Ollama instance required.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from mem0_mcp_selfhosted.reranker_ollama import (
    OllamaReranker,
    OllamaRerankerConfig,
    _extract_text,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_docs(*memories: str) -> list[dict]:
    return [{"id": str(i), "memory": m} for i, m in enumerate(memories)]


def _mock_response(results: list[dict]) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = {"results": results}
    resp.raise_for_status.return_value = None
    return resp


# ---------------------------------------------------------------------------
# OllamaRerankerConfig
# ---------------------------------------------------------------------------


class TestOllamaRerankerConfig:
    def test_defaults(self):
        cfg = OllamaRerankerConfig()
        assert cfg.ollama_base_url == "http://localhost:11434"
        assert cfg.timeout == 30.0
        assert cfg.model is None
        assert cfg.top_k is None

    def test_custom_values(self):
        cfg = OllamaRerankerConfig(
            model="mxbai-rerank-large-v1",
            ollama_base_url="http://gpu-host:11434",
            timeout=10.0,
            top_k=5,
        )
        assert cfg.model == "mxbai-rerank-large-v1"
        assert cfg.ollama_base_url == "http://gpu-host:11434"
        assert cfg.timeout == 10.0
        assert cfg.top_k == 5

    def test_model_from_base_class(self):
        cfg = OllamaRerankerConfig(model="bge-reranker-v2-m3")
        assert cfg.model == "bge-reranker-v2-m3"


# ---------------------------------------------------------------------------
# _extract_text
# ---------------------------------------------------------------------------


class TestExtractText:
    def test_memory_field(self):
        assert _extract_text({"memory": "hello"}) == "hello"

    def test_text_field_fallback(self):
        assert _extract_text({"text": "world"}) == "world"

    def test_content_field_fallback(self):
        assert _extract_text({"content": "foo"}) == "foo"

    def test_str_fallback(self):
        doc = {"other": "bar"}
        result = _extract_text(doc)
        assert "bar" in result

    def test_memory_takes_precedence_over_text(self):
        assert _extract_text({"memory": "mem", "text": "txt"}) == "mem"


# ---------------------------------------------------------------------------
# OllamaReranker.rerank — happy path
# ---------------------------------------------------------------------------


class TestOllamaRerankerRerank:
    def _make_reranker(self, **kwargs) -> OllamaReranker:
        cfg = OllamaRerankerConfig(model="bge-reranker-v2-m3", **kwargs)
        return OllamaReranker(cfg)

    def test_empty_documents_returns_empty_no_http(self):
        reranker = self._make_reranker()
        with patch("httpx.post") as mock_post:
            result = reranker.rerank("query", [])
        mock_post.assert_not_called()
        assert result == []

    def test_successful_rerank_sorted_descending(self):
        docs = _make_docs("alpha", "beta", "gamma")
        reranker = self._make_reranker()
        mock_resp = _mock_response([
            {"index": 2, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.5},
            {"index": 1, "relevance_score": 0.2},
        ])
        with patch("httpx.post", return_value=mock_resp):
            result = reranker.rerank("query", docs)
        assert [d["memory"] for d in result] == ["gamma", "alpha", "beta"]
        assert result[0]["rerank_score"] == 0.9
        assert result[1]["rerank_score"] == 0.5
        assert result[2]["rerank_score"] == 0.2

    def test_rerank_score_added_to_copy_not_original(self):
        docs = _make_docs("a", "b")
        reranker = self._make_reranker()
        mock_resp = _mock_response([
            {"index": 0, "relevance_score": 0.8},
            {"index": 1, "relevance_score": 0.3},
        ])
        with patch("httpx.post", return_value=mock_resp):
            result = reranker.rerank("q", docs)
        # Original docs are not mutated
        assert "rerank_score" not in docs[0]
        assert "rerank_score" in result[0]

    def test_top_k_from_argument(self):
        docs = _make_docs("a", "b", "c", "d", "e")
        reranker = self._make_reranker()
        mock_resp = _mock_response([
            {"index": i, "relevance_score": float(5 - i) / 5} for i in range(5)
        ])
        with patch("httpx.post", return_value=mock_resp):
            result = reranker.rerank("q", docs, top_k=3)
        assert len(result) == 3

    def test_top_k_from_config_when_not_passed(self):
        docs = _make_docs("a", "b", "c", "d", "e")
        reranker = self._make_reranker(top_k=2)
        mock_resp = _mock_response([
            {"index": i, "relevance_score": float(5 - i) / 5} for i in range(5)
        ])
        with patch("httpx.post", return_value=mock_resp):
            result = reranker.rerank("q", docs)
        assert len(result) == 2

    def test_correct_payload_sent_to_httpx(self):
        docs = _make_docs("fact one", "fact two")
        reranker = self._make_reranker()
        mock_resp = _mock_response([
            {"index": 0, "relevance_score": 0.7},
            {"index": 1, "relevance_score": 0.3},
        ])
        with patch("httpx.post", return_value=mock_resp) as mock_post:
            reranker.rerank("my query", docs)
        call_kwargs = mock_post.call_args
        url = call_kwargs[0][0] if call_kwargs[0] else call_kwargs.kwargs.get("url", call_kwargs[0][0])
        body = call_kwargs.kwargs["json"]
        assert "/api/rerank" in url
        assert body["model"] == "bge-reranker-v2-m3"
        assert body["query"] == "my query"
        assert body["documents"] == ["fact one", "fact two"]

    def test_trailing_slash_stripped_from_url(self):
        cfg = OllamaRerankerConfig(
            model="bge-reranker-v2-m3",
            ollama_base_url="http://localhost:11434/",
        )
        reranker = OllamaReranker(cfg)
        mock_resp = _mock_response([{"index": 0, "relevance_score": 0.5}])
        with patch("httpx.post", return_value=mock_resp) as mock_post:
            reranker.rerank("q", _make_docs("x"))
        url = mock_post.call_args[0][0]
        assert "//api" not in url
        assert url.endswith("/api/rerank")


# ---------------------------------------------------------------------------
# OllamaReranker.rerank — error fallback
# ---------------------------------------------------------------------------


class TestOllamaRerankerFallback:
    def _make_reranker(self) -> OllamaReranker:
        cfg = OllamaRerankerConfig(model="bge-reranker-v2-m3")
        return OllamaReranker(cfg)

    def test_http_error_returns_original_order(self):
        import httpx as _httpx

        docs = _make_docs("a", "b", "c")
        reranker = self._make_reranker()
        with patch("httpx.post", side_effect=_httpx.HTTPStatusError("err", request=MagicMock(), response=MagicMock())):
            result = reranker.rerank("q", docs)
        assert [d["memory"] for d in result] == ["a", "b", "c"]
        assert all("rerank_score" in d for d in result)

    def test_network_timeout_returns_original_order(self):
        import httpx as _httpx

        docs = _make_docs("x", "y")
        reranker = self._make_reranker()
        with patch("httpx.post", side_effect=_httpx.TimeoutException("timeout")):
            result = reranker.rerank("q", docs)
        assert [d["memory"] for d in result] == ["x", "y"]
        assert result[0]["rerank_score"] == 0.0

    def test_generic_exception_returns_original_order(self):
        docs = _make_docs("p", "q")
        reranker = self._make_reranker()
        with patch("httpx.post", side_effect=RuntimeError("unexpected")):
            result = reranker.rerank("q", docs)
        assert len(result) == 2

    def test_fallback_respects_top_k(self):
        docs = _make_docs("a", "b", "c", "d")
        reranker = self._make_reranker()
        with patch("httpx.post", side_effect=Exception("err")):
            result = reranker.rerank("q", docs, top_k=2)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Integration with RerankerFactory
# ---------------------------------------------------------------------------


class TestOllamaRerankerFactoryIntegration:
    def test_factory_create_after_registration(self):
        """After mutating provider_to_class, RerankerFactory.create returns OllamaReranker."""
        from mem0.utils.factory import RerankerFactory

        RerankerFactory.provider_to_class["ollama"] = (
            "mem0_mcp_selfhosted.reranker_ollama.OllamaReranker",
            OllamaRerankerConfig,
        )
        reranker = RerankerFactory.create("ollama", {"model": "bge-reranker-v2-m3"})
        assert isinstance(reranker, OllamaReranker)
        assert reranker._model == "bge-reranker-v2-m3"
