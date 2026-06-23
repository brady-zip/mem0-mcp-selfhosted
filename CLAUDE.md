# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## MCP Servers

- **mem0**: Persistent memory across sessions. At the start of each session, `search_memories` for relevant context before asking the user to re-explain anything. Use `add_memory` whenever you discover project architecture, coding conventions, debugging insights, key decisions, or user preferences. Use `update_memory` when prior context changes. Save information like: "This project uses PostgreSQL with Prisma", "Tests run with pytest -v", "Auth uses JWT validated in middleware". When in doubt, save it — future sessions benefit from over-remembering.

## Build & Test Commands

```bash
pip install -e ".[dev]"              # Install with dev dependencies
python3 -m pytest tests/unit/ -v     # Unit tests (mocked, no infra needed)
python3 -m pytest tests/contract/ -v # Contract tests (validates mem0ai internals)
python3 -m pytest tests/integration/ -v  # Integration tests (requires live Qdrant + Neo4j + Ollama)
python3 -m pytest tests/ -v          # All tests
python3 -m pytest tests/ -m "not integration" -v  # Skip integration
python3 -m pytest tests/unit/test_auth.py::TestIsOatToken -v  # Single test class
python3 -m pytest tests/unit/test_auth.py::TestIsOatToken::test_oat_token_detected -v  # Single test
```

## Architecture

Self-hosted MCP server using `mem0ai` as a library. 11 tools (9 memory + 2 graph), FastMCP orchestrator.

**Module roles:**
- `server.py` — FastMCP orchestrator, registers all tools + `memory_assistant` prompt
- `config.py` — Env vars → mem0ai `MemoryConfig` dict, handles all 5 graph LLM provider configs + the optional reranker block (`MEM0_RERANK_*`); embedder (`MEM0_EMBED_*`) and reranker are configured independently (either can be ZeroEntropy, neither, or both)
- `auth.py` — 3-tier token fallback: `MEM0_ANTHROPIC_TOKEN` → `~/.claude/.credentials.json` → `ANTHROPIC_API_KEY`
- `llm_anthropic.py` — Custom Anthropic provider registered with mem0ai's `LlmFactory`; handles OAT headers, structured outputs (JSON schema via `output_config`), and tool-call parsing
- `embed_zeroentropy.py` — Custom ZeroEntropy embedder (`zembed-1`); mem0ai has a `zero_entropy` reranker but no ZeroEntropy embedder, so this fills the gap. Registered via `server.register_embedders()` (mutates `EmbedderFactory.provider_to_class` — no `register_provider` on that factory). Maps mem0 `memory_action`→ZeroEntropy `input_type` (add/update→document, search→query) for asymmetric retrieval
- `llm_router.py` — `SplitModelGraphLLM` routes by tool name: extraction tools → Gemini, contradiction tools → Claude
- `helpers.py` — `_mem0_call()` error wrapper, `call_with_graph()` threading lock for per-call graph toggle (fast-paths when no graph store configured), `safe_bulk_delete()` iterates+deletes individually (never calls `memory.delete_all()`), `patch_graph_sanitizer()` monkey-patches mem0ai's relationship sanitizer for Neo4j compliance
- `graph_tools.py` — Direct Neo4j Cypher queries with lazy driver init
- `__init__.py` — Suppresses mem0ai telemetry before any imports

**Critical implementation details:**
- `memory.delete()` does NOT clean Neo4j nodes (mem0ai bug #3245) — `safe_bulk_delete()` explicitly calls `memory.graph.delete_all(filters)` after
- `memory.enable_graph` is mutable instance state — `call_with_graph()` holds a `threading.Lock` for the full duration of each Memory call (2-20s) when graph is active; fast-paths with no lock when `memory.graph is None` (enables concurrent multi-worktree use without Neo4j). Slow-path lock acquisition fails fast after `MEM0_LOCK_TIMEOUT_SECS` (default 60s)
- Contract tests (`tests/contract/`) validate mem0ai internal API assumptions — if these fail after a mem0ai upgrade, the code needs updating
- `Memory.update()` uses `data=` parameter, not `text=`
- Structured output support requires claude-opus-4/sonnet-4/haiku-4 models; older models fall back to JSON extraction
- Reranking is optional and provider-agnostic via mem0ai's pre-registered `RerankerFactory` providers — no custom module/registration. Two families: **local CrossEncoder** (`sentence_transformer`, `huggingface`) needs the `[rerank]` extra (`sentence-transformers`, pulls torch) and takes `MEM0_RERANK_DEVICE`; **hosted HTTP** (`zero_entropy` needs the `[zeroentropy]` extra, `cohere`) takes an api_key (`MEM0_RERANK_API_KEY`, falling back to `ZERO_ENTROPY_API_KEY`/`COHERE_API_KEY`) — `config.py` keeps device vs api_key on separate provider branches so a field is never sent to a schema that rejects it. `Memory.__init__` builds the reranker **eagerly**, so `hooks.py` force-disables it (`MEM0_RERANK_PROVIDER=""`): only the long-running server reranks, while short-lived passive-recall hook processes skip it (avoids a per-process cold model-load for local rerankers / a per-recall network round-trip for hosted ones). Interactive `search_memories` reranks by default when a reranker is configured (`mem.reranker is not None`); per-call `rerank=False` overrides
- ZeroEntropy is selectable independently as the **embedder** (`MEM0_EMBED_PROVIDER=zeroentropy`, model `zembed-1`, `[zeroentropy]` extra) and/or the **reranker** (`MEM0_RERANK_PROVIDER=zero_entropy`). The embedder is custom (`embed_zeroentropy.py`); the reranker is mem0ai's built-in. `zembed-1` is matryoshka — `MEM0_EMBED_DIMS` (default 2560) is forwarded as `dimensions` AND must equal the Qdrant collection width, so switching an existing store to it requires a fresh collection
- mem0ai's `sanitize_relationship_for_cypher()` has gaps (no hyphen handling, no leading-digit check) — `patch_graph_sanitizer()` wraps it at startup to ensure all relationship types match `^[a-zA-Z_][a-zA-Z0-9_]*$`
