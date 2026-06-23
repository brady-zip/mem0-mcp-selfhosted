"""Unit tests for the custom ZeroEntropy embedder (mocked SDK).

A fake ``zeroentropy`` module is injected into ``sys.modules`` so these run
whether or not the real package is installed. They assert the embedder's
contract with mem0ai: asymmetric ``input_type`` mapping, model/dimension
forwarding, batch order, and API-key resolution.
"""

from __future__ import annotations

import sys
import types

import pytest

from mem0.configs.embeddings.base import BaseEmbedderConfig


def _install_fake_zeroentropy(monkeypatch):
    """Inject a fake ``zeroentropy`` module; return (FakeZE, calls list)."""
    calls: list[dict] = []

    def embed(**kwargs):
        calls.append(kwargs)
        value = kwargs["input"]
        items = value if isinstance(value, list) else [value]
        # Deterministic fake vector per item; preserves input order.
        results = [types.SimpleNamespace(embedding=[float(i), float(len(t))]) for i, t in enumerate(items)]
        return types.SimpleNamespace(results=results)

    class FakeZeroEntropy:
        def __init__(self, api_key=None):
            self.api_key = api_key
            self.models = types.SimpleNamespace(embed=embed)

    mod = types.ModuleType("zeroentropy")
    mod.ZeroEntropy = FakeZeroEntropy
    monkeypatch.setitem(sys.modules, "zeroentropy", mod)
    return FakeZeroEntropy, calls


def _make(monkeypatch, **cfg_kwargs):
    """Build a ZeroEntropyEmbedding against the fake SDK."""
    _install_fake_zeroentropy(monkeypatch)
    from mem0_mcp_selfhosted.embed_zeroentropy import ZeroEntropyEmbedding

    cfg_kwargs.setdefault("api_key", "test-key")
    return ZeroEntropyEmbedding(BaseEmbedderConfig(**cfg_kwargs))


class TestZeroEntropyEmbedding:
    def test_default_model_is_zembed1(self, monkeypatch):
        emb = _make(monkeypatch)
        assert emb.config.model == "zembed-1"

    def test_embed_input_type_mapping(self, monkeypatch):
        _, calls = _install_fake_zeroentropy(monkeypatch)
        from mem0_mcp_selfhosted.embed_zeroentropy import ZeroEntropyEmbedding

        emb = ZeroEntropyEmbedding(BaseEmbedderConfig(api_key="k"))
        emb.embed("doc text", "add")
        emb.embed("upd text", "update")
        emb.embed("query text", "search")
        emb.embed("no action")  # None -> document
        assert [c["input_type"] for c in calls] == ["document", "document", "query", "document"]

    def test_embed_returns_first_vector(self, monkeypatch):
        emb = _make(monkeypatch)
        vec = emb.embed("hello", "search")
        assert vec == [0.0, 5.0]  # index 0, len("hello") == 5

    def test_dimensions_forwarded_only_when_set(self, monkeypatch):
        _, calls = _install_fake_zeroentropy(monkeypatch)
        from mem0_mcp_selfhosted.embed_zeroentropy import ZeroEntropyEmbedding

        with_dims = ZeroEntropyEmbedding(BaseEmbedderConfig(api_key="k", embedding_dims=1280))
        with_dims.embed("x", "search")
        assert calls[-1]["dimensions"] == 1280

        without = ZeroEntropyEmbedding(BaseEmbedderConfig(api_key="k"))
        without.embed("x", "search")
        assert "dimensions" not in calls[-1]

    def test_encoding_format_is_float(self, monkeypatch):
        _, calls = _install_fake_zeroentropy(monkeypatch)
        from mem0_mcp_selfhosted.embed_zeroentropy import ZeroEntropyEmbedding

        ZeroEntropyEmbedding(BaseEmbedderConfig(api_key="k")).embed("x", "add")
        assert calls[-1]["encoding_format"] == "float"

    def test_embed_batch_preserves_order_single_call(self, monkeypatch):
        _, calls = _install_fake_zeroentropy(monkeypatch)
        from mem0_mcp_selfhosted.embed_zeroentropy import ZeroEntropyEmbedding

        emb = ZeroEntropyEmbedding(BaseEmbedderConfig(api_key="k"))
        out = emb.embed_batch(["a", "bb", "ccc"], "add")
        # one API call for the whole batch
        assert len(calls) == 1
        assert calls[0]["input"] == ["a", "bb", "ccc"]
        # vectors come back in input order (fake encodes index + length)
        assert out == [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]

    def test_missing_api_key_raises(self, monkeypatch):
        _install_fake_zeroentropy(monkeypatch)
        monkeypatch.delenv("ZERO_ENTROPY_API_KEY", raising=False)
        from mem0_mcp_selfhosted.embed_zeroentropy import ZeroEntropyEmbedding

        with pytest.raises(ValueError, match="ZeroEntropy API key required"):
            ZeroEntropyEmbedding(BaseEmbedderConfig())

    def test_api_key_from_env(self, monkeypatch):
        _install_fake_zeroentropy(monkeypatch)
        monkeypatch.setenv("ZERO_ENTROPY_API_KEY", "env-key")
        from mem0_mcp_selfhosted.embed_zeroentropy import ZeroEntropyEmbedding

        emb = ZeroEntropyEmbedding(BaseEmbedderConfig())  # no api_key in config
        assert emb.client.api_key == "env-key"
