"""Claude Code session hooks for mem0-mcp-selfhosted.

Three entry points registered in pyproject.toml:
- mem0-hook-context  -> context_main()   (SessionStart)
- mem0-hook-stop     -> stop_main()      (Stop)
- mem0-install-hooks -> install_main()   (CLI installer)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# Load .env early so _get_user_id() sees MEM0_USER_ID even when it's
# called before _get_memory().  load_dotenv(override=False) is the
# default — it never clobbers values already in os.environ.
load_dotenv()

# Hooks write JSON responses to stdout — logging must go to stderr
# so it never corrupts the hook response channel.
logging.basicConfig(stream=sys.stderr, format="%(levelname)s %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared initialization
# ---------------------------------------------------------------------------

_memory = None

_MAX_MEMORIES = 20
_MIN_USER_LEN = 20
_MIN_ASSISTANT_LEN = 50
_MAX_CONTENT_LEN = 4000
_RECENT_WINDOW = 6  # last ~3 exchanges (user+assistant pairs)

_HANDOFF_DIR_ENV = "MEM0_HANDOFF_DIR"
_HANDOFF_RECALL_MAX = 4  # mem0 memories folded into the synthesis prompt
_HANDOFF_PREV_MAX = 2000  # chars of the previous handoff folded into synthesis


def _get_user_id() -> str:
    """Resolve user ID from MEM0_USER_ID env var, defaulting to ``'user'``."""
    return os.environ.get("MEM0_USER_ID", "user")


def _get_app_id() -> str | None:
    """Resolve the capture app_id (domain partition) from MEM0_APP_ID.

    Optional and backward-compatible: when unset (the generic / teammate
    case), capture writes no app_id and behaves exactly as before. When set
    (B's partitioned stack, where the hook wrapper exports the cwd-derived
    domain), stop_main tags the captured memory with ``metadata.app_id`` so it
    lands in the right bucket (e.g. ``evergreen`` vs ``general``).
    """
    val = os.environ.get("MEM0_APP_ID", "").strip()
    return val or None


def _get_recall_app_ids() -> list[str]:
    """Resolve the recall app_id filter list from MEM0_RECALL_APP_IDS.

    Optional and backward-compatible: a comma-separated list of app_ids to
    scope SessionStart recall to. When unset (generic case), recall filters on
    ``user_id`` only — unchanged behaviour. When set, context_main runs its
    multi-query search once per app_id (filtering on
    ``{"user_id": ..., "app_id": <id>}``) and merges/dedups the results, so an
    actor can recall across several domains (e.g. Hal reads ``hal-ops`` plus
    ``evergreen``).
    """
    raw = os.environ.get("MEM0_RECALL_APP_IDS", "")
    return [a.strip() for a in raw.split(",") if a.strip()]


def _get_memory():
    """Lazy-initialize and cache a mem0 Memory instance with graph disabled.

    Graph is force-disabled for speed — hooks must complete within the
    Claude Code timeout (15s for context, 30s for stop).  The instance
    is cached in a module global; since each hook invocation is a
    separate process, this only initializes once.
    """
    global _memory
    if _memory is not None:
        return _memory

    # Force graph off — the hard os.environ set overrides any .env value
    # that load_dotenv() loaded at module init.
    os.environ["MEM0_ENABLE_GRAPH"] = "false"

    # Force the reranker off for the same reason, plus a sharper one: the
    # reranker loads a CrossEncoder model in-process at Memory init (eager). Each
    # hook invocation is a fresh, short-lived process, so a configured reranker
    # would cold-load the model on every session start/stop — seconds of latency,
    # risking the hook timeout — for recall that never asks to be reranked.
    # Reranking belongs only in the long-running server.
    os.environ["MEM0_RERANK_PROVIDER"] = ""

    from mem0_mcp_selfhosted.config import build_config
    from mem0_mcp_selfhosted.server import register_embedders, register_providers

    config_dict, providers_info, _ = build_config()
    register_providers(providers_info)
    register_embedders()
    # patch_graph_sanitizer() skipped — graph is force-disabled in hooks,
    # so the relationship sanitizer modules are never invoked.

    from mem0 import Memory

    _memory = Memory.from_config(config_dict)
    return _memory


def _output(data: dict) -> None:
    """Print JSON to stdout (the hook response channel)."""
    print(json.dumps(data))


def _nonfatal() -> dict:
    """Return the standard non-fatal / no-op hook response.

    Must return a **fresh** dict each time — callers may mutate it
    (e.g. adding ``additionalContext``).
    """
    return {"continue": True, "suppressOutput": True}


# ---------------------------------------------------------------------------
# Context Hook  (SessionStart)
# ---------------------------------------------------------------------------


def _extract_results(raw) -> list[dict]:
    """Normalise mem0 search results to a flat list of dicts."""
    if isinstance(raw, dict):
        return raw.get("results", [])
    if isinstance(raw, list):
        return raw
    return []


def _filter_sets(user_id: str, recall_app_ids: list[str]) -> list[dict]:
    """Build the per-app_id filter sets for a scoped recall.

    When ``recall_app_ids`` is set, one filter per app_id (so mem0 2.x's
    metadata app_id match applies per domain); otherwise a single user_id
    filter (backward-compatible generic behaviour).
    """
    if recall_app_ids:
        return [{"user_id": user_id, "app_id": app_id} for app_id in recall_app_ids]
    return [{"user_id": user_id}]


def _search_scoped(
    mem,
    queries: list[str],
    user_id: str,
    recall_app_ids: list[str],
    top_k: int = 15,
    max_total: int = _MAX_MEMORIES,
    threshold: float | None = None,
) -> list[dict]:
    """Run each query against each app_id filter set, merged/deduped by id."""
    seen_ids: set[str] = set()
    out: list[dict] = []
    for filters in _filter_sets(user_id, recall_app_ids):
        for query in queries:
            kwargs: dict = {"query": query, "filters": filters, "top_k": top_k}
            if threshold is not None:
                kwargs["threshold"] = threshold
            for r in _extract_results(mem.search(**kwargs)):
                mid = r.get("id")
                if mid and mid not in seen_ids:
                    seen_ids.add(mid)
                    out.append(r)
    return out[:max_total]


def _emit_additional_context(event_name: str, context: str) -> None:
    """Emit a hookSpecificOutput.additionalContext response for *event_name*.

    This is the canonical shape for UserPromptSubmit / PreToolUse context
    injection (matches Claude Code's documented hook output).
    """
    _output({
        "hookSpecificOutput": {
            "hookEventName": event_name,
            "additionalContext": context,
        }
    })


def context_main() -> None:
    """SessionStart hook: inject cross-session memories as additionalContext."""
    try:
        hook_input = json.loads(sys.stdin.read())
        cwd = hook_input.get("cwd", "")
        project_name = Path(cwd).name if cwd else "project"
        if not project_name:
            project_name = "project"
        user_id = _get_user_id()
        recall_app_ids = _get_recall_app_ids()

        mem = _get_memory()

        # Multi-query, multi-app_id search merged/deduped by id. NOTE(fork):
        # mem0ai 2.x search(query, *, top_k, filters, ...) takes entity scopes
        # inside `filters` (not top-level) and uses `top_k`; app_id is matched
        # against the value persisted in the Qdrant payload.
        all_memories = _search_scoped(
            mem,
            queries=[
                f"project context, architecture, conventions for {project_name}",
                f"recent session summary, decisions, key changes for {project_name}",
            ],
            user_id=user_id,
            recall_app_ids=recall_app_ids,
            top_k=15,
            max_total=_MAX_MEMORIES,
        )

        if not all_memories:
            _output(_nonfatal())
            return

        # Format as numbered lines
        lines = ["# mem0 Cross-Session Memory\n"]
        for i, m in enumerate(all_memories, 1):
            text = m.get("memory", m.get("text", ""))
            lines.append(f"{i}. {text}")

        response = _nonfatal()
        response["additionalContext"] = "\n".join(lines)
        _output(response)

    except Exception:
        logger.debug("context_main failed", exc_info=True)
        _output(_nonfatal())


# ---------------------------------------------------------------------------
# Stop Hook
# ---------------------------------------------------------------------------


def _extract_content(content) -> str:
    """Extract plain text from a transcript content field.

    Claude Code transcripts use content blocks:
    ``[{"type": "text", "text": "..."}]``
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        ]
        return " ".join(parts)
    return ""


def _read_recent_messages(transcript_path: str) -> list[tuple[str, str]]:
    """Read recent user/assistant messages from a JSONL transcript.

    Returns up to ``_RECENT_WINDOW`` ``(role, content)`` tuples in
    chronological order.  Uses a bounded deque so memory stays O(1)
    regardless of transcript length (which can reach ~900 KB).
    Content is truncated during parsing to avoid holding large
    assistant responses (tool results, file reads) in memory.
    """
    messages: deque[tuple[str, str]] = deque(maxlen=_RECENT_WINDOW)

    with open(transcript_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Skip subagent (sidechain) turns — internal to a Task run.
            if entry.get("isSidechain"):
                continue

            # Claude Code transcripts wrap the message:
            #   {"type": "user"|"assistant", "message": {"role", "content"}}
            # plus many non-message line types (mode, attachment, snapshots…).
            # Older/synthetic transcripts use a flat {"role", "content"}.
            # Support both by reading from .message when present, else top-level.
            msg = entry.get("message")
            src = msg if isinstance(msg, dict) else entry
            role = src.get("role", "")
            if role not in ("user", "assistant"):
                continue
            # _extract_content keeps only text blocks, so tool_use / tool_result
            # entries naturally collapse to "" and are skipped below.
            content = _extract_content(src.get("content", ""))[:_MAX_CONTENT_LEN]
            if content:
                messages.append((role, content))

    return list(messages)


def _handoff_dir() -> Path:
    """Resolve the directory handoff files are written to.

    Override with ``MEM0_HANDOFF_DIR``; otherwise XDG data home (defaulting to
    ``~/.local/share``) under ``mem0-brady/handoffs``. Kept alongside the
    plugin's other data so a SessionStart pointer can find the latest one.
    """
    override = os.environ.get(_HANDOFF_DIR_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_DATA_HOME", "").strip()
    root = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
    return root / "mem0-brady" / "handoffs"


def _handoff_path_for(cwd: str, project_name: str) -> Path:
    """Deterministic handoff path keyed by cwd (NOT session_id).

    The hook-payload session_id is not stable across hook types within one
    Claude session, so a file keyed on it can't be found on resume. cwd is
    stable and also separates worktrees that share a project name (the path
    hash disambiguates them).
    """
    key = cwd or project_name
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", project_name).strip("-") or "project"
    return _handoff_dir() / f"{safe}-{digest}.md"


def _git_status_block(cwd: str) -> str:
    """Best-effort ``git status -sb`` (truncated) for the handoff appendix.

    Returns "" when cwd isn't a git repo or git is unavailable — never raises.
    """
    if not cwd:
        return ""
    try:
        out = subprocess.run(
            ["git", "-C", cwd, "status", "-sb"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if out.returncode != 0:
        return ""
    return "\n".join(out.stdout.strip().splitlines()[:15])


def _read_previous_handoff(cwd: str, project_name: str) -> str:
    """Return the prose recap of the existing handoff about to be overwritten.

    Each meaningful turn overwrites the handoff at ``_handoff_path_for``; folding
    the prior one back into synthesis makes the next recap a *continuation* (it
    can carry forward goals/watch-outs the last few transcript turns no longer
    mention) instead of a fresh take on a short window. Strips the auto-written
    HTML comment + metadata header and the git-status appendix so only the recap
    prose is fed back, truncated to ``_HANDOFF_PREV_MAX`` chars. Returns "" when
    no prior handoff exists or it can't be read — never raises.
    """
    try:
        path = _handoff_path_for(cwd, project_name)
        if not path.is_file():
            return ""
        text = path.read_text(encoding="utf-8")
    except OSError:
        logger.debug("reading previous handoff failed", exc_info=True)
        return ""

    # Drop the git-status appendix (everything from its "### git status" rule).
    marker = text.find("### git status")
    if marker != -1:
        text = text[:marker].rstrip().rstrip("-").rstrip()

    # Start at the **Goal** recap, skipping the HTML comment + metadata header.
    goal = text.find("**Goal**")
    if goal != -1:
        text = text[goal:]

    return text.strip()[:_HANDOFF_PREV_MAX]


# ---------------------------------------------------------------------------
# Workstreams  (multi-session work spanning worktrees; see the plugin skill
# /mem0-brady:workstream). A session is tagged with a workstream by an active
# pointer keyed on session_id; when present, the Stop / PreCompact hooks fold
# the workstream's overview into the handoff recap and bake a re-activation
# call into the file so the workstream rides the handoff chain forward. Every
# helper fails open — an untagged session (no pointer) behaves exactly as before.
# ---------------------------------------------------------------------------

_WORKSTREAM_DIR_ENV = "MEM0_WORKSTREAM_DIR"
_WORKSTREAM_OVERVIEW_MAX = 1500  # chars of the workstream doc folded into synthesis


def _workstream_dir() -> Path:
    """Resolve the directory workstream docs + active pointers live under.

    Override with ``MEM0_WORKSTREAM_DIR``; otherwise XDG data home (defaulting
    to ``~/.local/share``) under ``mem0-brady/workstreams`` — alongside the
    handoffs the workstream's pieces reference.
    """
    override = os.environ.get(_WORKSTREAM_DIR_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_DATA_HOME", "").strip()
    root = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
    return root / "mem0-brady" / "workstreams"


def _active_workstream(session_id: str) -> dict | None:
    """Return the active-workstream pointer for *session_id*, or None.

    The pointer (``<workstream_dir>/active/<session_id>.json``) is written by
    the /mem0-brady:workstream skill when a session is activated. A session
    with no pointer is untagged. Fail-open: any error returns None.
    """
    if not session_id:
        return None
    try:
        path = _workstream_dir() / "active" / f"{session_id}.json"
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.debug("reading active workstream failed", exc_info=True)
        return None
    return data if isinstance(data, dict) and data.get("slug") else None


def _workstream_overview(slug: str) -> str:
    """Return the workstream doc's Goal + Pieces prose for synthesis context.

    Reads ``<workstream_dir>/<slug>.md`` from the ``## Goal`` heading onward
    (skipping the HTML comment + metadata header), truncated to
    ``_WORKSTREAM_OVERVIEW_MAX`` chars. The Pieces list is references only —
    each piece's current state lives in its own handoff — so this stays a
    compact overview. "" on any failure.
    """
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", slug or "").strip("-")
    if not safe:
        return ""
    try:
        path = _workstream_dir() / f"{safe}.md"
        if not path.is_file():
            return ""
        text = path.read_text(encoding="utf-8")
    except OSError:
        logger.debug("reading workstream overview failed", exc_info=True)
        return ""
    goal = text.find("## Goal")
    if goal != -1:
        text = text[goal:]
    return text.strip()[:_WORKSTREAM_OVERVIEW_MAX]


def _synthesize_handoff(
    mem,
    recent: list[tuple[str, str]],
    project_name: str,
    cwd: str = "",
    workstream: dict | None = None,
) -> str:
    """One LLM call turning the recent transcript + mem0 recall into a recap.

    Uses mem0's already-configured chat LLM (provider-agnostic via
    ``mem.llm.generate_response``). Reads the previous handoff (via *cwd*) so the
    new recap builds on it rather than starting cold. Returns "" on any failure
    so the caller can skip writing the file.
    """
    convo = "\n\n".join(
        f"[{'User' if r == 'user' else 'Assistant'}]: {c}" for r, c in recent
    )

    # Build on the prior handoff (the one this run will overwrite, read before
    # _write_handoff overwrites it) so the recap is a continuation.
    previous_block = (
        _read_previous_handoff(cwd, project_name)
        or "(none — first handoff for this project)"
    )

    # "augment with mem0": fold a scoped recall into the synthesis context so
    # the recap is informed by long-term memory, not just the last few turns.
    recalled_block = "(none)"
    try:
        mems = _search_scoped(
            mem,
            queries=[
                "session state, current task, work in progress, recent decisions"
            ],
            user_id=_get_user_id(),
            recall_app_ids=_get_recall_app_ids(),
            top_k=_HANDOFF_RECALL_MAX,
            max_total=_HANDOFF_RECALL_MAX,
        )
        if mems:
            recalled_block = "\n".join(
                f"- {m.get('memory', m.get('text', ''))}" for m in mems
            )
    except Exception:
        logger.debug("handoff recall failed", exc_info=True)

    # When the session is tagged with a workstream, fold its overview (goal +
    # the index of sibling pieces) into the synthesis so this per-cwd recap is
    # situated within the larger, multi-session objective. Per-piece current
    # state stays in each piece's own handoff — referenced, never inlined here.
    workstream_block = ""
    workstream_hint = ""
    if workstream and workstream.get("slug"):
        overview = _workstream_overview(workstream["slug"])
        if overview:
            workstream_block = (
                f"## Active workstream '{workstream['slug']}' "
                f"(overarching, multi-session context):\n{overview}\n\n"
            )
            workstream_hint = (
                " This session is part of the workstream above — keep the Goal "
                "consistent with its overarching objective, and do not restate "
                "sibling pieces' state (that lives in their own handoffs)."
            )

    prompt = (
        "You are writing a terse resume handoff so a future agent (or the same "
        "user returning to a cold context) can pick up a coding session "
        "immediately. Be concrete: name files, PR numbers, identifiers.\n\n"
        f"{workstream_block}"
        f"## Previous handoff (the recap you are updating):\n{previous_block}\n\n"
        f"## Recent conversation (oldest first), project '{project_name}':\n"
        f"{convo}\n\n"
        f"## Relevant long-term memory:\n{recalled_block}\n\n"
        "Treat the previous handoff as prior state: carry forward goals and "
        "watch-outs that still hold, update State/Next from the recent "
        "conversation, and drop anything now done. Do not copy it verbatim."
        f"{workstream_hint}\n\n"
        "Write markdown under ~180 words, omitting any section that does not "
        "apply, with these headers:\n"
        "- **Goal** — the overarching objective in one sentence.\n"
        "- **State** — what is done/shipped so far (bullets).\n"
        "- **Next** — the immediate next step(s).\n"
        "- **Watch out** — gotchas, blockers, or pending user decisions.\n"
        "Start directly with the **Goal** line — no document title, no "
        "preamble, no closing remarks."
    )

    try:
        resp = mem.llm.generate_response(
            messages=[{"role": "user", "content": prompt}]
        )
    except Exception:
        logger.debug("handoff synthesis failed", exc_info=True)
        return ""

    # generate_response returns a str on the no-tools path across providers,
    # but be defensive about a dict-shaped return.
    if isinstance(resp, dict):
        resp = resp.get("content") or resp.get("text") or ""
    return (resp or "").strip()


def _write_handoff(
    mem,
    recent: list[tuple[str, str]],
    project_name: str,
    cwd: str,
    source: str,
    workstream: dict | None = None,
) -> Path | None:
    """Synthesize and write the handoff markdown. Returns its path or None.

    Fail-open: any error is swallowed (logged) and returns None so the Stop /
    PreCompact response is never broken. When *workstream* is set, the recap is
    synthesized with the workstream overview and a deterministic re-activation
    call is baked into the file so a resuming session re-tags itself.
    """
    try:
        body = _synthesize_handoff(mem, recent, project_name, cwd, workstream)
        if not body:
            return None

        path = _handoff_path_for(cwd, project_name)
        path.parent.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        slug = workstream.get("slug") if workstream else None
        parts = [
            "<!-- mem0-brady handoff (auto-written; overwritten each meaningful turn) -->",
            f"# Handoff — {project_name}",
            "",
            f"- generated: {ts} (source: {source})",
            f"- cwd: `{cwd}`",
        ]
        if slug:
            # Point the metadata at the workstream doc itself (referenceable),
            # falling back to the conventional path if the pointer omitted it.
            doc_path = workstream.get("doc_path") or str(_workstream_dir() / f"{slug}.md")
            parts.append(f"- workstream: `{slug}` → `{doc_path}`")
        parts += ["", body]
        # Bake the re-activation call in deterministically (not via the LLM, so
        # it is never dropped): a session resuming from this handoff runs the
        # skill, re-tags itself, and pulls the workstream overview + sibling work.
        if slug:
            parts += [
                "",
                f"**Workstream** — part of `{slug}`. To resume with the full "
                f"overarching context (goal + sibling work), run "
                f"`/mem0-brady:workstream {slug}` at the start of the session.",
            ]
        git_block = _git_status_block(cwd)
        if git_block:
            parts += ["", "---", "### git status", "```", git_block, "```"]
        path.write_text("\n".join(parts) + "\n", encoding="utf-8")
        return path
    except Exception:
        logger.debug("_write_handoff failed", exc_info=True)
        return None


def _capture_summary(
    transcript_path: str,
    session_id: str,
    project_name: str,
    source: str,
    cwd: str = "",
) -> Path | None:
    """Read the transcript, store a summary in mem0, and write a handoff file.

    Shared by the Stop and PreCompact hooks. Skips short transcripts. Tags the
    mem0 memory with ``source`` and (when MEM0_APP_ID is set) the domain
    ``app_id``. Returns the handoff file path (for the caller's systemMessage)
    or None when nothing meaningful was captured / written.
    """
    if not transcript_path or not Path(transcript_path).is_file():
        return None

    recent = _read_recent_messages(transcript_path)

    # Skip short sessions — AND means we save when *either* side contributed
    # meaningful content (e.g. short question + long answer).
    user_total = sum(len(c) for r, c in recent if r == "user")
    asst_total = sum(len(c) for r, c in recent if r == "assistant")
    if user_total < _MIN_USER_LEN and asst_total < _MIN_ASSISTANT_LEN:
        return None

    exchanges = []
    for role, content in recent:
        label = "User" if role == "user" else "Assistant"
        exchanges.append(f"[{label}]: {content}")

    summary = (
        f"Session summary for project '{project_name}':\n\n"
        + "\n\n".join(exchanges)
        + "\n\n"
        "Extract key decisions, solutions found, patterns discovered, "
        "configuration changes, and important context for future sessions."
    )

    metadata: dict = {"source": source, "session_id": session_id}
    # When MEM0_APP_ID is set, tag with the domain partition so the memory
    # lands in the right bucket and is filterable on recall. When unset, no
    # app_id is written (backward-compatible generic case).
    app_id = _get_app_id()
    if app_id:
        metadata["app_id"] = app_id

    # When the session is tagged with a workstream (active pointer keyed on
    # session_id), tag the captured memory with workstream_id so passive recall
    # and /digest can filter by workstream, and pass it into the handoff so the
    # recap is workstream-aware and carries the re-activation call.
    workstream = _active_workstream(session_id)
    if workstream:
        metadata["workstream_id"] = workstream["slug"]

    mem = _get_memory()
    mem.add(
        messages=[{"role": "user", "content": summary}],
        user_id=_get_user_id(),
        infer=True,
        metadata=metadata,
    )

    # Resume-recap handoff (reuses the messages already read + the same mem
    # instance). Fail-open inside _write_handoff.
    return _write_handoff(mem, recent, project_name, cwd, source, workstream)


def stop_main() -> None:
    """Stop hook: save a session summary to mem0 (source=session-stop-hook)."""
    try:
        hook_input = json.loads(sys.stdin.read())

        # Infinite-loop guard: Claude Code sets this when re-entering
        if hook_input.get("stop_hook_active"):
            _output(_nonfatal())
            return

        cwd = hook_input.get("cwd", "")
        project_name = Path(cwd).name if cwd else "project"
        if not project_name:
            project_name = "project"

        handoff = _capture_summary(
            hook_input.get("transcript_path", ""),
            hook_input.get("session_id", ""),
            project_name,
            "session-stop-hook",
            cwd,
        )
        resp = _nonfatal()
        if handoff:
            resp["systemMessage"] = f"handoff → {handoff}"
        _output(resp)

    except Exception:
        logger.debug("stop_main failed", exc_info=True)
        _output(_nonfatal())


# ---------------------------------------------------------------------------
# PreCompact Hook
# ---------------------------------------------------------------------------


def precompact_main() -> None:
    """PreCompact hook: capture a session-state summary before compaction.

    Same capture path as Stop but tagged ``source=pre-compact-hook`` and fired
    mid-session (when context is about to be lost to compaction), so a resume
    after compaction can recall what was in flight. Silent (no output beyond
    the non-fatal ack).
    """
    try:
        hook_input = json.loads(sys.stdin.read())
        cwd = hook_input.get("cwd", "")
        project_name = Path(cwd).name if cwd else "project"
        if not project_name:
            project_name = "project"
        handoff = _capture_summary(
            hook_input.get("transcript_path", ""),
            hook_input.get("session_id", ""),
            project_name,
            "pre-compact-hook",
            cwd,
        )
        resp = _nonfatal()
        if handoff:
            resp["systemMessage"] = f"handoff → {handoff}"
        _output(resp)
    except Exception:
        logger.debug("precompact_main failed", exc_info=True)
        _output(_nonfatal())


# ---------------------------------------------------------------------------
# UserPromptSubmit Hook
# ---------------------------------------------------------------------------

_PROMPT_MIN_LEN = 20
_RESUME_RE = re.compile(
    r"(where (did )?(we|i) (leave|left) off|continue (from )?(where|last)|"
    r"what were we (working|doing)|pick up where|resume|"
    r"what.?s the (current|latest) (state|status)|catch me up|where are we)",
    re.IGNORECASE,
)


def prompt_main() -> None:
    """UserPromptSubmit hook: search-decision rubric + resume-intent recall.

    Recall and prose only — this hook never CAPTURES (no blind auto-capture),
    so it adds no duplication. On resume-intent it pre-searches mem0 (scoped to
    MEM0_RECALL_APP_IDS) and injects the recovered context; once per session it
    injects a short rubric steering the agent to search mem0 itself.
    """
    try:
        hook_input = json.loads(sys.stdin.read())
        prompt = hook_input.get("prompt", "") or ""
        if len(prompt) < _PROMPT_MIN_LEN:
            _output(_nonfatal())
            return

        session_id = hook_input.get("session_id", "") or "default"
        user_id = _get_user_id()
        recall_app_ids = _get_recall_app_ids()
        parts: list[str] = []

        # Resume-intent → actually pre-search mem0 and inject.
        if _RESUME_RE.search(prompt):
            mem = _get_memory()
            mems = _search_scoped(
                mem,
                queries=[
                    "session state, current task, work in progress",
                    "recent decisions, key changes, and learnings",
                ],
                user_id=user_id,
                recall_app_ids=recall_app_ids,
                top_k=5,
                max_total=6,
            )
            if mems:
                lines = ["# mem0 — recovered context for resuming:"]
                for i, m in enumerate(mems, 1):
                    lines.append(f"{i}. {m.get('memory', m.get('text', ''))}")
                parts.append("\n".join(lines))

        # Once-per-session search rubric (flag file keyed on session id).
        flag = os.path.join(tempfile.gettempdir(), f"mem0_rubric_{session_id}")
        if not os.path.exists(flag):
            parts.append(
                "Mem0 recall: when a prompt references past work, a decision, an "
                "error, or a non-trivial task, search mem0 first "
                "(mcp__mem0__search_memories, scoped to this session's app_id) "
                "before answering or asking the user."
            )
            try:
                open(flag, "w").close()
            except OSError:
                pass

        if parts:
            _emit_additional_context("UserPromptSubmit", "\n\n".join(parts))
        else:
            _output(_nonfatal())

    except Exception:
        logger.debug("prompt_main failed", exc_info=True)
        _output(_nonfatal())


# ---------------------------------------------------------------------------
# File-Context Hook  (PreToolUse: Read)
# ---------------------------------------------------------------------------

_FILE_CONTEXT_MIN_BYTES = 1500
_FILE_CONTEXT_MAX = 5


def file_context_main() -> None:
    """PreToolUse(Read) hook: inject prior mem0 context about the file.

    Gates files >= _FILE_CONTEXT_MIN_BYTES, searches mem0 (app_id-scoped) for
    the file path, and injects a compact list of prior memories. Recall only —
    never blocks the Read, never captures.
    """
    try:
        hook_input = json.loads(sys.stdin.read())
        tool_input = hook_input.get("tool_input", {}) or {}
        file_path = tool_input.get("file_path") or tool_input.get("path") or ""
        cwd = hook_input.get("cwd", "") or os.getcwd()
        if not file_path:
            _output(_nonfatal())
            return

        p = Path(file_path)
        if not p.is_absolute():
            p = Path(cwd) / p
        try:
            if not p.is_file() or p.stat().st_size < _FILE_CONTEXT_MIN_BYTES:
                _output(_nonfatal())
                return
        except OSError:
            _output(_nonfatal())
            return

        try:
            rel = os.path.relpath(str(p), cwd)
        except ValueError:
            rel = str(p)
        basename = p.name
        query = f"{rel} {basename}" if rel != basename else rel

        mem = _get_memory()
        mems = _search_scoped(
            mem,
            queries=[query],
            user_id=_get_user_id(),
            recall_app_ids=_get_recall_app_ids(),
            top_k=_FILE_CONTEXT_MAX,
            max_total=_FILE_CONTEXT_MAX,
            threshold=0.3,
        )
        if not mems:
            _output(_nonfatal())
            return

        lines = [f"Prior mem0 context for `{rel}` ({len(mems)} memories):"]
        for m in mems:
            text = (m.get("memory", m.get("text", "")) or "")[:150]
            text = text.replace("\n", " ").strip()
            lines.append(f"- {text}")
        _emit_additional_context("PreToolUse", "\n".join(lines))

    except Exception:
        logger.debug("file_context_main failed", exc_info=True)
        _output(_nonfatal())


# ---------------------------------------------------------------------------
# Install-Hooks CLI
# ---------------------------------------------------------------------------

_HOOK_CONTEXT_CMD = "mem0-hook-context"
_HOOK_STOP_CMD = "mem0-hook-stop"


def _has_hook(hooks_list: list, command: str) -> bool:
    """Check if a hook with the given command already exists.

    Searches both the current nested format and the legacy flat format::

        Nested:  [{"matcher": "...", "hooks": [{"type": "command", "command": "..."}]}]
        Legacy:  [{"matcher": "...", "command": "..."}]
    """
    for group in hooks_list:
        if not isinstance(group, dict):
            continue
        # Current nested format
        for handler in group.get("hooks") or []:
            if isinstance(handler, dict) and handler.get("command") == command:
                return True
        # Legacy flat format (pre-nested schema)
        if group.get("command") == command:
            return True
    return False


_HANDLER_KEYS = {"command", "timeout"}
_GROUP_KEYS = {"matcher"}


def _migrate_legacy_hooks(hooks_list: list) -> list:
    """Convert legacy flat-format hooks to the nested format.

    Flat entries (``{"command": "...", "timeout": ...}``) are converted to
    nested format (``{"hooks": [{"type": "command", ...}]}``).  Already-nested
    entries are kept as-is.  Non-dict entries are discarded.  Unknown keys are
    forwarded to preserve any extra properties the user may have set.
    """
    migrated = []
    for group in hooks_list:
        if not isinstance(group, dict):
            continue
        if "hooks" in group:
            # Already in nested format
            migrated.append(group)
        elif "command" in group:
            # Legacy flat format — convert, forwarding unknown keys to
            # group level so no user data is silently dropped.
            handler: dict = {"type": "command"}
            new_group: dict = {}
            for k, v in group.items():
                if k in _HANDLER_KEYS:
                    handler[k] = v
                elif k in _GROUP_KEYS:
                    new_group[k] = v
                else:
                    new_group[k] = v
            new_group["hooks"] = [handler]
            migrated.append(new_group)
        else:
            # Unknown format — preserve as-is
            migrated.append(group)
    return migrated


def install_main() -> None:
    """CLI: install mem0 hooks into .claude/settings.json."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="mem0-install-hooks",
        description="Install mem0 session hooks for Claude Code",
    )
    parser.add_argument(
        "--global",
        dest="global_install",
        action="store_true",
        help="Install to ~/.claude/settings.json instead of project directory",
    )
    parser.add_argument(
        "--project-dir",
        default=None,
        help="Project directory (defaults to CWD)",
    )
    args = parser.parse_args()

    if args.global_install:
        settings_dir = Path.home() / ".claude"
    else:
        project_dir = Path(args.project_dir) if args.project_dir else Path.cwd()
        if not project_dir.is_dir():
            print(
                f"Error: project directory does not exist: {project_dir}",
                file=sys.stderr,
            )
            sys.exit(1)
        settings_dir = project_dir / ".claude"

    settings_dir.mkdir(parents=True, exist_ok=True)
    settings_path = settings_dir / "settings.json"

    # Read existing settings (preserve everything)
    if settings_path.exists():
        try:
            with open(settings_path, encoding="utf-8") as f:
                settings = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error: {settings_path} contains invalid JSON: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        settings = {}

    if not isinstance(settings.get("hooks"), dict):
        settings["hooks"] = {}

    hooks = settings["hooks"]

    # Migrate any legacy flat-format hooks to nested format
    for event_key in ("SessionStart", "Stop"):
        if isinstance(hooks.get(event_key), list):
            hooks[event_key] = _migrate_legacy_hooks(hooks[event_key])

    installed: list[str] = []
    skipped: list[str] = []

    # --- SessionStart hook ---
    if not isinstance(hooks.get("SessionStart"), list):
        hooks["SessionStart"] = []
    if _has_hook(hooks["SessionStart"], _HOOK_CONTEXT_CMD):
        skipped.append(f"SessionStart ({_HOOK_CONTEXT_CMD})")
    else:
        hooks["SessionStart"].append({
            "matcher": "startup|compact",
            "hooks": [
                {
                    "type": "command",
                    "command": _HOOK_CONTEXT_CMD,
                    "timeout": 15000,
                }
            ],
        })
        installed.append(f"SessionStart ({_HOOK_CONTEXT_CMD})")

    # --- Stop hook ---
    if not isinstance(hooks.get("Stop"), list):
        hooks["Stop"] = []
    if _has_hook(hooks["Stop"], _HOOK_STOP_CMD):
        skipped.append(f"Stop ({_HOOK_STOP_CMD})")
    else:
        hooks["Stop"].append({
            "hooks": [
                {
                    "type": "command",
                    "command": _HOOK_STOP_CMD,
                    "timeout": 30000,
                }
            ],
        })
        installed.append(f"Stop ({_HOOK_STOP_CMD})")

    # Atomic write: temp file + rename avoids truncated settings on crash
    fd, tmp_path = tempfile.mkstemp(dir=str(settings_dir), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
            f.write("\n")
        os.replace(tmp_path, str(settings_path))
    except BaseException:
        os.unlink(tmp_path)
        raise

    # Report
    for hook in installed:
        print(f"Installed: {hook}")
    for hook in skipped:
        print(f"Already installed: {hook}")
    print(f"Settings: {settings_path}")
