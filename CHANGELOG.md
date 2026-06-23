# CHANGELOG


## v0.7.1 (2026-06-23)

### Bug Fixes

- **concurrency**: Lock-free fast path when no graph store (v0.7.1)
  ([`4ec3096`](https://github.com/brady-zip/mem0-mcp-selfhosted/commit/4ec3096718ce6d5a4ca92531b1670806d90ef332))

call_with_graph() now skips the global threading.Lock entirely when memory.graph is None (the common
  Qdrant-only, no-Neo4j setup on mem0ai 2.x). In that configuration every caller writes
  enable_graph=False identically, so there is no flag to race on. The lock previously serialized
  every memory op (each held 2-20s, since Memory.add() blocks on concurrent.futures.wait()),
  bottlenecking the multiple worktrees that share the launchd MCP server.

The graph-active path keeps the lock but acquires with a configurable timeout
  (MEM0_LOCK_TIMEOUT_SECS, default 60s), raising RuntimeError on timeout instead of hanging.

Tests: adds TestNoGraphFastPath (concurrency, all-observe-false, lock

timeout). Reranker from PR #1 intentionally omitted (stock Ollama has no /api/rerank endpoint).


## v0.7.0 (2026-06-18)

### Features

- Resume-handoff file on Stop/PreCompact + real-transcript parse fix (v0.7.0)
  ([`8ce3ad2`](https://github.com/brady-zip/mem0-mcp-selfhosted/commit/8ce3ad2a6683fdd2bf911b13e0bbbc3851c30c62))

Stop/PreCompact hooks now also write a synthesized resume-handoff markdown file (one
  mem.llm.generate_response call, augmented with a scoped mem0 recall) to a cwd-keyed path under
  ~/.local/share/mem0-brady/handoffs/, and surface that path to the user via the hook response
  systemMessage. Path is keyed by cwd, not the unstable session_id, so a fresh session can locate it
  on resume.

Also fixes _read_recent_messages, which only understood a flat top-level role/content shape and
  returned zero messages on real (wrapped) Claude Code transcripts — silently no-opping both the new
  handoff and the pre-existing mem.add session-summary capture. It now reads
  message.role/message.content, supports the flat shape, and skips sidechain/tool-result noise.


## v0.6.1 (2026-06-12)

### Bug Fixes

- Default missing app_id to 'general' not retired 'meta' (v0.6.1)
  ([`c7b3f95`](https://github.com/brady-zip/mem0-mcp-selfhosted/commit/c7b3f95e6312bb783df1bf3c9d05e8bde433f319))

A write that omits app_id and bypasses the caller-side guards landed in 'meta', a bucket no actor
  reads — orphaning it from recall. Default to 'general' (a real, readable domain). Update app_id
  field descriptions to evergreen/general/hal-ops.


## v0.6.0 (2026-06-11)

### Features

- Richer lifecycle hooks — prompt/file-context/pre-compact entrypoints (v0.6.0)
  ([`9e128d3`](https://github.com/brady-zip/mem0-mcp-selfhosted/commit/9e128d3487be1bed6eb552f8981855154c574552))

mem0-hook-prompt (UserPromptSubmit): once-per-session search rubric + resume-intent recall; never
  captures. mem0-hook-filecontext (PreToolUse Read): inject prior mem0 context for files >=1500B.
  mem0-hook-precompact (PreCompact): capture session-state summary before compaction
  (source=pre-compact-hook), sharing _capture_summary with Stop. Factor
  _search_scoped/_emit_additional_context. Deliberately omits blind per-N-message auto-capture to
  avoid duplication. Bump 0.6.0.


## v0.5.0 (2026-06-11)

### Features

- App_id-aware auto-capture/recall hooks (MEM0_APP_ID / MEM0_RECALL_APP_IDS)
  ([`7c7b31c`](https://github.com/brady-zip/mem0-mcp-selfhosted/commit/7c7b31cdf440534d9e1e84d2e9a1d216b3e67c38))

stop_main tags captured memories with metadata.app_id when MEM0_APP_ID is set; context_main scopes
  recall per app_id when MEM0_RECALL_APP_IDS is set (searched per-id, merged/deduped). Both default
  off — generic behaviour unchanged. Lets the same fork serve a partitioned personal stack and
  generic teammate use. Bump 0.5.0.


## v0.4.0 (2026-06-11)

### Bug Fixes

- Add .python-version for Glama uv sync compatibility
  ([`e4d1f09`](https://github.com/brady-zip/mem0-mcp-selfhosted/commit/e4d1f09008652a84ed1340db9372f621b8ffa785))

Pin Python 3.12 so uv sync resolves the correct interpreter in Glama's Docker build environment
  instead of picking up Debian's externally-managed Python 3.11.

- Cache-bust Glama badge URL to force fresh camo proxy fetch
  ([`205ecf9`](https://github.com/brady-zip/mem0-mcp-selfhosted/commit/205ecf9a6d8d95f23fa0d8fa27826e3348ab0728))

- Mem0ai 2.x search API in SessionStart context hook
  ([`4aaa4b8`](https://github.com/brady-zip/mem0-mcp-selfhosted/commit/4aaa4b8e4919b37e47ca0770d8121a35703ba4cd))

context_main still used the 1.x search(user_id=, limit=) signature, which mem0ai 2.x rejects (entity
  scopes belong in filters=, and the arg is top_k=). The exception was swallowed by the hook's
  fail-open except, so auto-recall silently did nothing. Switch to search(filters={user_id},
  top_k=), matching the server's already-2.x search_memories tool. stop_main's add() was already
  2.x-compatible (add still accepts user_id/infer).

- Update hooks to nested format for Claude Code schema compatibility
  ([`2f86dee`](https://github.com/brady-zip/mem0-mcp-selfhosted/commit/2f86dee99c3fa73220270b721c1621881beea655))

Migrate hook installer from the deprecated flat format to the current nested schema (matcher group
  -> hooks array -> handler objects). Add legacy format detection and auto-migration so existing
  users upgrading do not end up with duplicate or broken entries.

- Use NEO4J_DATABASE env var instead of config dict for non-default database
  ([`74e1188`](https://github.com/brady-zip/mem0-mcp-selfhosted/commit/74e1188d38154846ec8b12602fde1d757197873b))

mem0ai's graph_memory.py passes config as positional args to Neo4jGraph() where pos 3 is `token`,
  not `database`. Setting database in the config dict causes it to land in the token parameter,
  resulting in AuthenticationError. Use NEO4J_DATABASE env var which langchain_neo4j reads via
  get_from_dict_or_env().

Upstream: mem0ai #3906, #3981, #4085 (none merged)

Resolves: PAR-57

- **ci**: Use angular parser compatible with PSR v9.15.2
  ([`b5bc6ab`](https://github.com/brady-zip/mem0-mcp-selfhosted/commit/b5bc6ab45edff26f07fc73774c7e0c57d22cb40d))

The v9 GitHub Action does not recognize "conventional" parser name (v10+ only). Reverts to "angular"
  and changelog.changelog_file format.

### Chores

- Remove Dockerfile (Glama generates its own)
  ([`33f2f1d`](https://github.com/brady-zip/mem0-mcp-selfhosted/commit/33f2f1d25bdb1e4c85617e90b21a72c48fc9c2a2))

Glama's admin page generates a Dockerfile from configuration fields rather than using the repo's
  Dockerfile. No other Docker deployment workflow exists, so the file is unused.

### Continuous Integration

- Add python-semantic-release configuration and GitHub Actions workflow
  ([`2473ee4`](https://github.com/brady-zip/mem0-mcp-selfhosted/commit/2473ee4ec9c0db90b2bb412d3714caae7dc41498))

Automated versioning via Conventional Commits analysis, changelog generation, git tagging
  (v{version}), and GitHub Release creation on push to main.

### Documentation

- Clarify hooks and CLAUDE.md as complementary layers
  ([`94f29dc`](https://github.com/brady-zip/mem0-mcp-selfhosted/commit/94f29dca52582ee18ce9ae256fc06d8cf1adab30))

Update README to explain that hooks (automated memory at session boundaries) and CLAUDE.md
  (behavioral instructions for mid-session engagement) work best together rather than as
  alternatives.

### Features

- Add Claude Code session hooks for cross-session memory
  ([`113df26`](https://github.com/brady-zip/mem0-mcp-selfhosted/commit/113df2678b05091dd0acffa2776c755d4c380644))

Add SessionStart and Stop hooks that give Claude Code automatic cross-session memory without
  requiring CLAUDE.md rules or manual tool calls.

- SessionStart hook (mem0-hook-context): searches mem0 with multi-query strategy, deduplicates by
  ID, injects formatted memories as additionalContext on startup and compact events - Stop hook
  (mem0-hook-stop): reads last ~3 exchanges from JSONL transcript via bounded deque, saves session
  summary to mem0 with infer=True for atomic fact extraction - CLI installer (mem0-install-hooks):
  patches .claude/settings.json with idempotent hook entries, supports --global and --project-dir -
  Graph force-disabled in hooks to stay within 15s/30s timeout budgets - Atomic settings.json write
  via tempfile + os.replace - 43 unit tests covering protocol, edge cases, and error handling - 6
  integration tests against live Qdrant + Ollama infrastructure - README updated with hooks
  documentation, architecture diagram, and test structure

- Lazy Memory init + Glama submission packaging
  ([`c6f2b76`](https://github.com/brady-zip/mem0-mcp-selfhosted/commit/c6f2b76aa7fc1f243c86fbcd941825ef7861b539))

Defer Memory.from_config() to the first tool call via _ensure_memory(), allowing the MCP server to
  respond to initialize/tools/list without live Qdrant/Neo4j/Ollama. This unblocks Glama's
  Docker-based inspection pipeline which builds and runs the container in an ephemeral sandbox.

Add LICENSE (MIT), glama.json, Dockerfile, and Glama badge in README.

- Openai provider + embedded on-disk Qdrant (MEM0_QDRANT_PATH)
  ([`58c1ada`](https://github.com/brady-zip/mem0-mcp-selfhosted/commit/58c1ada870fb935cc18a6728e29bb0bb32e6c759))

Lands the OpenAI LLM/embedder provider (one OPENAI_API_KEY for both) and app_id partitioning that
  powered the local stack, plus a new embedded Qdrant mode: when MEM0_QDRANT_PATH is set, mem0ai's
  QdrantConfig runs QdrantClient(path=...) for a server-less, persistent on-disk store (no Qdrant
  server / Docker / :6333). Enables a fully self-contained, multi-session-safe stack behind a single
  launchd agent.

Documents OPENAI_API_KEY and MEM0_QDRANT_PATH in .env.example and pins uv.lock for reproducible 'uv
  tool install' from git.
