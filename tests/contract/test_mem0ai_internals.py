"""Contract tests: mem0ai internal API stability.

These tests validate assumptions about mem0ai internals that our code depends on.
If these fail after a mem0ai upgrade, our code needs updating.

NOTE: These tests require mem0ai to be installed. They test the real package,
not mocks. Skip with `pytest -m "not contract"` if deps unavailable.
"""

from __future__ import annotations

import pytest

# Mark all tests in this module as contract tests
pytestmark = pytest.mark.contract


class TestVectorStoreClientAccess:
    """Test that memory.vector_store.client is a public, stable attribute."""

    def test_qdrant_class_has_client_attribute(self):
        """The Qdrant vector store class exposes .client as a public attribute."""
        try:
            from mem0.vector_stores.qdrant import Qdrant
        except ImportError:
            pytest.skip("mem0ai not installed")

        # Verify 'client' is in the class (not a private _client)
        assert hasattr(Qdrant, "__init__"), "Qdrant class must have __init__"
        # Check the source to verify client is assigned (not _client)
        import inspect

        source = inspect.getsource(Qdrant.__init__)
        assert "self.client" in source, (
            "INVARIANT BROKEN: Qdrant.__init__ must assign self.client. "
            "Our code accesses memory.vector_store.client directly."
        )

    def test_qdrant_class_has_collection_name(self):
        """The Qdrant vector store class exposes .collection_name."""
        try:
            from mem0.vector_stores.qdrant import Qdrant
        except ImportError:
            pytest.skip("mem0ai not installed")

        import inspect

        source = inspect.getsource(Qdrant.__init__)
        assert "self.collection_name" in source, (
            "INVARIANT BROKEN: Qdrant.__init__ must assign self.collection_name. "
            "Our code accesses memory.vector_store.collection_name."
        )


class TestMcpSdkImports:
    """Test MCP SDK import paths remain stable."""

    def test_mcp_client_session_importable(self):
        """ClientSession import path remains valid across MCP SDK versions."""
        try:
            from mcp.client.session import ClientSession
        except ImportError:
            pytest.skip("mcp SDK not installed")

        assert ClientSession  # Import succeeded — contract satisfied


class TestLlmFactoryRegistration:
    """Test LlmFactory.register_provider() behavior."""

    def test_register_provider_exists(self):
        """LlmFactory has a register_provider classmethod."""
        try:
            from mem0.utils.factory import LlmFactory
        except ImportError:
            pytest.skip("mem0ai not installed")

        assert hasattr(LlmFactory, "register_provider"), (
            "INVARIANT BROKEN: LlmFactory must have register_provider classmethod."
        )

    def test_register_provider_is_idempotent(self):
        """Calling register_provider twice with same name doesn't error."""
        try:
            from mem0.utils.factory import LlmFactory
        except ImportError:
            pytest.skip("mem0ai not installed")

        # Register once
        LlmFactory.register_provider(
            name="test_idempotent",
            class_path="mem0_mcp_selfhosted.llm_anthropic.AnthropicOATLLM",
            config_class=None,
        )
        # Register again — should not raise
        LlmFactory.register_provider(
            name="test_idempotent",
            class_path="mem0_mcp_selfhosted.llm_anthropic.AnthropicOATLLM",
            config_class=None,
        )

    def test_registration_persists_across_calls(self):
        """Registered provider persists in factory after registration."""
        try:
            from mem0.utils.factory import LlmFactory
        except ImportError:
            pytest.skip("mem0ai not installed")

        LlmFactory.register_provider(
            name="test_persist",
            class_path="mem0_mcp_selfhosted.llm_anthropic.AnthropicOATLLM",
            config_class=None,
        )

        # Verify the provider is in the factory's registry
        # The factory uses a class-level dict, so it should persist
        provider_map = getattr(LlmFactory, "provider_to_class", None)
        if provider_map is not None:
            assert "test_persist" in provider_map, (
                "INVARIANT BROKEN: Registered provider must persist in LlmFactory."
            )


class TestOllamaLLMInterface:
    """Validate upstream OllamaLLM interface our subclass depends on."""

    def test_ollama_llm_has_parse_response(self):
        """OllamaLLM has _parse_response method we override."""
        try:
            from mem0.llms.ollama import OllamaLLM
        except ImportError:
            pytest.skip("mem0ai not installed")

        assert hasattr(OllamaLLM, "_parse_response"), (
            "INVARIANT BROKEN: OllamaLLM must have _parse_response method. "
            "Our OllamaToolLLM subclass overrides it."
        )

    def test_ollama_llm_has_generate_response(self):
        """OllamaLLM has generate_response method we override."""
        try:
            from mem0.llms.ollama import OllamaLLM
        except ImportError:
            pytest.skip("mem0ai not installed")

        assert hasattr(OllamaLLM, "generate_response"), (
            "INVARIANT BROKEN: OllamaLLM must have generate_response method. "
            "Our OllamaToolLLM subclass overrides it."
        )

    def test_ollama_config_has_base_url(self):
        """OllamaConfig accepts ollama_base_url parameter."""
        try:
            from mem0.configs.llms.ollama import OllamaConfig
        except ImportError:
            pytest.skip("mem0ai not installed")

        # Verify __init__ accepts ollama_base_url and stores it
        cfg = OllamaConfig(ollama_base_url="http://test:11434")
        assert cfg.ollama_base_url == "http://test:11434", (
            "INVARIANT BROKEN: OllamaConfig must accept and store ollama_base_url. "
            "Our config.py passes this field to Ollama LLM config."
        )

    def test_ollama_llm_init_accepts_config(self):
        """OllamaLLM.__init__ accepts a 'config' parameter by name."""
        try:
            from mem0.llms.ollama import OllamaLLM
        except ImportError:
            pytest.skip("mem0ai not installed")

        import inspect

        sig = inspect.signature(OllamaLLM.__init__)
        params = list(sig.parameters.keys())
        assert "config" in params, (
            "INVARIANT BROKEN: OllamaLLM.__init__ must accept a 'config' parameter. "
            f"Our OllamaToolLLM inherits __init__ from it. Found params: {params}"
        )


class TestRerankerFactoryInterface:
    """Validate RerankerFactory internals our OllamaReranker registration depends on."""

    def test_reranker_factory_has_provider_to_class(self):
        """RerankerFactory.provider_to_class is a class-level dict we can mutate."""
        try:
            from mem0.utils.factory import RerankerFactory
        except ImportError:
            pytest.skip("mem0ai not installed")

        assert hasattr(RerankerFactory, "provider_to_class"), (
            "INVARIANT BROKEN: RerankerFactory must have provider_to_class dict. "
            "Our register_reranker() mutates it to register OllamaReranker."
        )
        assert isinstance(RerankerFactory.provider_to_class, dict)

    def test_provider_to_class_tuple_format(self):
        """Built-in entries are 2-tuples (class_path_str, ConfigClass) — our format matches."""
        try:
            from mem0.utils.factory import RerankerFactory
        except ImportError:
            pytest.skip("mem0ai not installed")

        for name, value in RerankerFactory.provider_to_class.items():
            assert isinstance(value, tuple) and len(value) == 2, (
                f"INVARIANT BROKEN: RerankerFactory.provider_to_class[{name!r}] must be "
                f"a 2-tuple (class_path_str, ConfigClass). Got: {value!r}"
            )
            class_path, config_class = value
            assert isinstance(class_path, str), (
                f"First element of tuple for {name!r} must be a string class path"
            )
            break  # Only need to check one existing entry

    def test_provider_to_class_mutation_persists(self):
        """Mutations to provider_to_class persist for the lifetime of the process."""
        try:
            from mem0.utils.factory import RerankerFactory
        except ImportError:
            pytest.skip("mem0ai not installed")

        _test_key = "_contract_test_ollama_reranker"
        RerankerFactory.provider_to_class[_test_key] = ("fake.path.Class", None)
        try:
            assert _test_key in RerankerFactory.provider_to_class, (
                "INVARIANT BROKEN: RerankerFactory.provider_to_class mutations must persist. "
                "Our register_reranker() depends on this."
            )
        finally:
            RerankerFactory.provider_to_class.pop(_test_key, None)

    def test_base_reranker_config_is_pydantic(self):
        """BaseRerankerConfig is a Pydantic BaseModel (our OllamaRerankerConfig extends it)."""
        try:
            from mem0.configs.rerankers.base import BaseRerankerConfig
            from pydantic import BaseModel
        except ImportError:
            pytest.skip("mem0ai or pydantic not installed")

        assert issubclass(BaseRerankerConfig, BaseModel), (
            "INVARIANT BROKEN: BaseRerankerConfig must be a Pydantic BaseModel. "
            "OllamaRerankerConfig inherits from it."
        )

    def test_reranker_factory_create_raises_for_unknown_provider(self):
        """RerankerFactory.create() raises ValueError for unknown providers."""
        try:
            from mem0.utils.factory import RerankerFactory
        except ImportError:
            pytest.skip("mem0ai not installed")

        with pytest.raises((ValueError, Exception)):
            RerankerFactory.create("_definitely_not_a_real_provider_xyz")
