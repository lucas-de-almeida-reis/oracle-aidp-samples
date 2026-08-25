"""
Multi-MCP Chat Agent — session-variable auth + context management.

Sibling of ../agent.py. Same idea (one chat agent fronting ADB Select AI,
OAC and OIC MCP servers) with the product-intended AUTH model, PLUS the
conversation context-management machinery ported back from ../agent.py.

────────────────────────────────────────────────────────────────────────
AUTH — this agent NEVER mints a token
────────────────────────────────────────────────────────────────────────
The bearer for each MCP server is supplied as an AIDP *session variable*
and resolved by aidputils at call time:

    auth = {"authType": "BEARER_TOKEN",
            "token": "{{sessionvariables.cred.mcp.<server>.bearer}}"}

- DEV : store a hand-made bearer in the Agent Studio "Variables" tab. It
        arrives on each request as a session variable.
- PROD: the calling application passes the bearer per request. Token
        generation stays OUTSIDE AIDP.

aidputils' BearerTokenAuthStrategy resolves `{{sessionvariables...}}` from
the per-request context (chat_context.session_context_var, populated by
pre_tool_setup) on EVERY tool call and on discovery — so the tool
catalogue can be discovered once and shared while each call still uses
that one user's own bearer.

────────────────────────────────────────────────────────────────────────
TOOLS — discovered live, lazily, on the first invoke
────────────────────────────────────────────────────────────────────────
We use aidputils' MCP client to discover (client.get_tools) and
build_structured_tools_from_allowed_mcp_tools to turn the discovered
schemas into executable, per-call-authenticated StructuredTools.

────────────────────────────────────────────────────────────────────────
CONTEXT MANAGEMENT (ported from ../agent.py)
────────────────────────────────────────────────────────────────────────
Even with correct auth, OCI's *generic* provider flattens tool schemas and
the model's tool-use reliability degrades after a long tool-calling
history ("long-context laziness"). This sample addresses that with:

  • MAX_TURNS_KEPT — hard cap on conversation history by user-turn count
    (the single most effective, deterministic lever). On by default.
  • llm.max_context_tokens — optional token-budget trim, sized from the
    real prompt_tokens OCI reports on the previous turn.
  • orphan tool_call healing — removes AIMessage(tool_calls) left without a
    matching ToolMessage (e.g. AIDP suspended the process mid-call), which
    otherwise makes providers reject the whole history.
  • /reset — wipes conversation state on demand.
  • context-overflow + INVALID_CHAT_HISTORY recovery.

NOT ported: the token-refresh / MCP-rebuild reactive retry from the parent
— that was specific to in-agent token minting. Here the bearer is resolved
per call from the session variable, so a 401 means "caller must supply a
valid bearer", not "rebuild and retry".

These state operations rely on the compiled-graph state API
(agent.aget_state / aupdate_state) — the same API create_react_agent
exposes; create_agent is built on the same LangGraph base. If a runtime
doesn't expose it, every state op degrades gracefully to a no-op (logged),
leaving the agent functional but without trimming.

Config: ./config.yaml (copy from config.sample.yaml).
"""

import asyncio
import json
import logging
import textwrap  # noqa: F401  (kept for parity with sibling samples)
from pathlib import Path
from typing import AsyncGenerator, Dict, Union

from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, HumanMessage, RemoveMessage

# Conversation-history trimming helpers. Available in langchain_core ≥0.3.
# Fall back to None if the AIDP runtime ships an older version — the trim
# feature then no-ops gracefully.
try:
    from langchain_core.messages.utils import (
        count_tokens_approximately as _lc_count_tokens,
        trim_messages as _lc_trim_messages,
    )
except ImportError:
    _lc_count_tokens = None
    _lc_trim_messages = None

from aidputils.agents.toolkit.agent_helper import (
    init_oci_llm,
    pre_invoke_setup,
    pre_tool_setup,
    post_tool_setup,
    parse_stream_response,
)
from aidputils.agents.toolkit.tool_helper import (
    build_structured_tools_from_allowed_mcp_tools,
)
from aidputils.agents.toolkit.configs import OCIAIConf
from aidputils.agents.tools.mcp.mcp_service import get_mcp_client

AGENT_ID = "multi-mcp-session-variables-agent"
logger = logging.getLogger(AGENT_ID)


# ╔══════════════════════════════════════════════════════════════╗
# ║   MAX_TURNS_KEPT — hard cap on conversation history          ║
# ║                                                              ║
# ║   When > 0, keep only the last N user-message turns before   ║
# ║   each LLM call. Older turns are removed; AIMessage(tool_     ║
# ║   calls)+ToolMessage pairs stay together (cut at a Human     ║
# ║   boundary). Set 0 to disable.                               ║
# ║                                                              ║
# ║   This is the most effective lever against tool-use          ║
# ║   degradation from long history + generic-provider schema    ║
# ║   flattening, and it's deterministic.                        ║
# ╚══════════════════════════════════════════════════════════════╝
MAX_TURNS_KEPT = 5

# Budget headroom reserved for the upcoming HumanMessage (not yet in state
# when _trim_history runs). ~200 tokens ≈ 800 chars of input.
NEXT_HUMAN_SLACK_TOKENS = 200


# ╔══════════════════════════════════════════════════════════════╗
# ║   Config loading                                             ║
# ╚══════════════════════════════════════════════════════════════╝

_CFG_INIT_ERROR: str | None = None
_CFG_PATHS_TRIED: list[str] = []
CFG: dict = {}


def _find_config_yaml() -> Path | None:
    direct = [
        Path(__file__).resolve().parent / "config.yaml",
        Path.cwd() / "config.yaml",
        Path.cwd().parent / "config.yaml",
        Path.home() / "config.yaml",
    ]
    for c in direct:
        _CFG_PATHS_TRIED.append(str(c))
        if c.exists():
            return c
    for base in (Path(__file__).resolve().parent, Path.cwd(), Path.cwd().parent):
        if not base.exists():
            continue
        try:
            for p in base.rglob("config.yaml"):
                _CFG_PATHS_TRIED.append(f"(rglob) {p}")
                return p
        except Exception:
            pass
    return None


try:
    import yaml  # PyYAML — pre-installed in the AIDP runtime
    found = _find_config_yaml()
    if found is None:
        raise FileNotFoundError(
            "config.yaml not found. Searched:\n"
            + "\n".join(f"  • {p}" for p in _CFG_PATHS_TRIED)
            + "\n\nUpload config.yaml next to agent.py in the AIDP project."
        )
    CFG = yaml.safe_load(found.read_text()) or {}
    logger.info("Loaded config.yaml from %s", found)
except Exception:
    import traceback as _tb
    _CFG_INIT_ERROR = _tb.format_exc()


# ── LLM config ──────────────────────────────────────────────────
LLM_COMPARTMENT_ID = CFG.get("llm", {}).get("compartment_id", "")
LLM_REGION         = CFG.get("llm", {}).get("region", "us-ashburn-1")
LLM_MODEL_ID       = CFG.get("llm", {}).get("model_id", "")
LLM_MODEL_ARGS     = CFG.get("llm", {}).get("model_args", {}) or {}

# Token-budget trim (optional). 0 == disabled (turn cap alone applies).
# When > 0, oldest turns are dropped so that
#   max_context_tokens − response_reserve_tokens − system − tool_defs
# isn't exceeded. MAX_TURNS_KEPT and this can both be active; whichever
# cuts more wins on a given turn.
LLM_MAX_CONTEXT_TOKENS      = int(CFG.get("llm", {}).get("max_context_tokens", 0) or 0)
LLM_RESPONSE_RESERVE_TOKENS = int(CFG.get("llm", {}).get("response_reserve_tokens", 1024) or 1024)

# ── Integrations ────────────────────────────────────────────────
_RAW_INTEGRATIONS = CFG.get("integrations", {}) or {}
INTEGRATIONS: dict[str, dict] = {}
for _name in ("adb", "oac", "oic"):
    _ic = _RAW_INTEGRATIONS.get(_name) or {}
    if _ic:
        INTEGRATIONS[_name] = _ic


def _enabled_integrations() -> list[tuple[str, dict]]:
    return [(n, ic) for n, ic in INTEGRATIONS.items() if ic.get("enabled")]


# ╔══════════════════════════════════════════════════════════════╗
# ║   session_config — declares the credential session variables  ║
# ╚══════════════════════════════════════════════════════════════╝
session_config: dict = {"variables": {}}
for _name, _ic in _enabled_integrations():
    _var = _ic.get("bearer_session_variable")
    if _var:
        session_config["variables"][_var] = {
            "name": _var,
            "isRequired": True,
            "shouldLog": False,
            "isSystem": True,
        }


# ╔══════════════════════════════════════════════════════════════╗
# ║   System prompt — adapts to enabled integrations             ║
# ╚══════════════════════════════════════════════════════════════╝
SYSTEM_PROMPT_INTRO = (
    "You are a helpful data assistant with access to tools loaded from "
    "one or more MCP servers. Users ask data questions in plain English; "
    "you answer them by calling the right tool."
)
_TOOL_DESCRIPTIONS = {
    "adb": (
        "  - ADB Select AI tool: natural-language-to-SQL over an Oracle "
        "Autonomous Database. Pass questions in natural language, fully "
        "qualified with table / column names when the user names them."
    ),
    "oac": (
        "  - Oracle Analytics Cloud (OAC) tools: query OAC subject areas / "
        "datasets with governance-aware Logical SQL. Discover and describe "
        "before executing so you use exact table / column names."
    ),
    "oic": (
        "  - Oracle Integration Cloud (OIC) tools: invoke OIC integrations "
        "exposed by the project's MCP server."
    ),
}
_RULES = (
    "\nHow to respond — follow these strictly:\n"
    "  - ACT, don't announce. NEVER say you 'will' call a tool, 'will query "
    "now', 'let me check', 'consultando agora', etc. and then stop. If you "
    "need data, call the tool IMMEDIATELY in this same turn, then answer. "
    "Every turn must end with EITHER a tool call OR a complete answer — never "
    "a promise to act later.\n"
    "  - ALWAYS include the data. After a tool returns rows, your reply MUST "
    "contain the actual values (a compact Markdown table or list). NEVER say "
    "'here are the results' / 'the data is shown' / 'apareceu?' without "
    "actually including the rows. NEVER reply with empty or whitespace-only "
    "content.\n"
    "  - If the user says the data didn't appear ('cadê os dados', 'não "
    "apareceu nada', 'eai?', 'e aí?'), it means your previous turn failed to "
    "call the tool or omitted the rows. Immediately call the correct tool NOW "
    "and present the data — do not just repeat tool/dataset lists.\n"
    "  - Reuse what you already discovered. Do NOT re-list datasets/tools or "
    "re-run discovery you already ran earlier in this conversation; build on "
    "the prior results.\n"
    "  - Pick the right tool. To 'see rows' of a table/dataset, run the QUERY "
    "tool (e.g. execute_logical_sql) — not discovery. Use discovery only when "
    "you don't yet know the exact dataset/column names.\n"
    "  - For follow-ups, rewrite using prior-turn context — MCP tools see "
    "only the prompt you pass them.\n"
    "  - Oracle SQL dialect. Every connected system is an Oracle product, so "
    "any SQL you write or ask a tool to run MUST be Oracle-compatible: use "
    "Oracle syntax — FETCH FIRST n ROWS ONLY (never LIMIT / TOP), SYSDATE / "
    "CURRENT_DATE, NVL or COALESCE, TO_DATE / TO_CHAR for dates, string "
    "concatenation with || . For OAC, use its Logical SQL. Never emit "
    "MySQL / PostgreSQL / SQL Server-specific syntax.\n"
    "  - Never invent numbers. If a tool returns an ERROR (e.g. 502 Bad "
    "Gateway, timeout, empty / no result) or 0 rows, report the failure "
    "plainly — name the tool/table and STOP. NEVER fabricate, estimate, or "
    "extrapolate a value from a failed call or a partial sample (e.g. do NOT "
    "say 'N clientes com base em uma amostra de N'). A failed or sampled call "
    "is not a total.\n"
    "  - Be complete but concise: a short plain-language summary AND the "
    "supporting rows. If results are long, prefer aggregates or a small row "
    "cap (Oracle: FETCH FIRST n ROWS ONLY) rather than truncating mid-table."
)


def _build_system_prompt() -> str:
    parts = [SYSTEM_PROMPT_INTRO]
    for name, _ in _enabled_integrations():
        if name in _TOOL_DESCRIPTIONS:
            parts.append(_TOOL_DESCRIPTIONS[name])
    return "\n".join(parts) + "\n" + _RULES


AGENT_SYSTEM_PROMPT = _build_system_prompt()


# ── LLM connection config ───────────────────────────────────────
llm_conf = OCIAIConf(
    model_provider="generic",
    compartment_id=LLM_COMPARTMENT_ID,
    model_args=LLM_MODEL_ARGS,
    endpoint=f"https://inference.generativeai.{LLM_REGION}.oci.oraclecloud.com",
    model_id=LLM_MODEL_ID,
    guardrails_config={},
)


def _bearer_auth(integration_cfg: dict) -> dict:
    var = integration_cfg.get("bearer_session_variable")
    return {"authType": "BEARER_TOKEN", "token": "{{" + str(var) + "}}"}


# ╔══════════════════════════════════════════════════════════════╗
# ║   Error classifiers (module-level, recurse ExceptionGroup)   ║
# ╚══════════════════════════════════════════════════════════════╝

def _is_invalid_history(e: BaseException) -> bool:
    """LangGraph's INVALID_CHAT_HISTORY — an AIMessage with tool_calls but
    no matching ToolMessage (typical when AIDP suspends mid-tool-call)."""
    sub = getattr(e, "exceptions", None)
    if sub:
        return any(_is_invalid_history(s) for s in sub)
    msg = str(e)
    return (
        "INVALID_CHAT_HISTORY" in msg
        or "tool_calls that do not have a corresponding ToolMessage" in msg
    )


def _is_context_overflow_error(e: BaseException) -> bool:
    """Model context-window / prompt-token-limit error (OCI surfaces these
    as 400s with varied wording). Broad on purpose: a false positive just
    triggers a cheap last-turn clear; a false negative leaves the agent
    stuck."""
    sub = getattr(e, "exceptions", None)
    if sub:
        return any(_is_context_overflow_error(s) for s in sub)
    msg = str(e).lower()
    return any(s in msg for s in (
        "maximum context length", "max context length",
        "context_length_exceeded", "context length exceeded",
        "token limit exceeded", "exceeds the maximum",
        "input is too long", "prompt is too long",
        "request too large", "too many tokens", "context window",
    ))


# ╔══════════════════════════════════════════════════════════════╗
# ║   AgentBasic — what AIDP discovers                           ║
# ╚══════════════════════════════════════════════════════════════╝
class AgentBasic:
    def __init__(self) -> None:
        self.llm = None
        # The compiled agent (create_agent → LangGraph compiled graph).
        # Stub (no tools) after setup, rebuilt with discovered tools on the
        # first invoke. State ops (aget_state/aupdate_state) run against it.
        self.agent = None

        self._tools_loaded = False
        self._last_tool_names: list[str] = []
        self._load_lock = None
        self._setup_error: str | None = None
        self._missing_bearers: list[tuple[str, str]] = []
        self._discovery_errors: list[tuple[str, str]] = []

        # ── Trim / usage state ──────────────────────────────────
        # Approx token cost of the bound tool defs (sizes the budget on the
        # first turn, before OCI reports a real prompt_tokens).
        self._tools_token_estimate: int = 0
        # Ground truth captured from the previous successful response.
        self._last_prompt_tokens: int | None = None
        self._measured_overhead_tokens: int | None = None
        # Set by _trim_history when even one turn can't fit the budget.
        self._trim_dropped_critical_context: bool = False
        # Outcome of the last hard-clear, surfaced by /reset.
        self._last_clear_strategy: str = ""

    # ── Setup (sync) ────────────────────────────────────────────
    def setup(self) -> None:
        if _CFG_INIT_ERROR:
            self._setup_error = (
                "config.yaml could not be loaded.\n\n"
                f"{_CFG_INIT_ERROR}\n\n"
                "Upload config.yaml next to agent.py in the AIDP project."
            )
            return
        if not _enabled_integrations():
            self._setup_error = (
                "No integration enabled in config.yaml. Set 'enabled: true' "
                "under integrations.adb / .oac / .oic."
            )
            return

        self.llm = init_oci_llm(llm_conf)
        self.agent = self._make_agent([])
        logger.info(
            "Setup complete. Enabled: %s. Tools discover lazily on first "
            "invoke. MAX_TURNS_KEPT=%d, max_context_tokens=%d.",
            [n for n, _ in _enabled_integrations()],
            MAX_TURNS_KEPT, LLM_MAX_CONTEXT_TOKENS,
        )

    def _make_agent(self, tools: list):
        kwargs = dict(
            name=AGENT_ID,
            model=self.llm,
            tools=tools,
            system_prompt=AGENT_SYSTEM_PROMPT,
            debug=True,
        )
        if checkpointer:
            kwargs["checkpointer"] = checkpointer
        return create_agent(**kwargs)

    # ── Live discovery + tool build ─────────────────────────────
    async def _discover_and_build_tools(self) -> list:
        all_tools: list = []
        missing: list[tuple[str, str]] = []
        errors: list[tuple[str, str]] = []

        for name, ic in _enabled_integrations():
            server_name = ic.get("server_name")
            endpoint = ic.get("endpoint")
            transport = ic.get("transport", "streamable_http")
            headers = ic.get("headers") or {}
            auth = _bearer_auth(ic)

            try:
                client = get_mcp_client(
                    server_name=server_name,
                    server_url=endpoint,
                    auth=auth,
                    transport=transport,
                    custom_headers=headers,
                )
                mcp_tools = await client.get_tools(server_name=server_name)
            except KeyError:
                missing.append((name, ic.get("bearer_session_variable")))
                logger.warning(
                    "No bearer for '%s' (session variable '%s' missing) — "
                    "skipping discovery.", name, ic.get("bearer_session_variable"),
                )
                continue
            except Exception as e:
                errors.append((name, str(e)))
                logger.exception("Discovery failed for '%s'", name)
                continue

            allowed = []
            for t in mcp_tools:
                tname = getattr(t, "name", None)
                if not tname:
                    continue
                allowed.append({
                    "tool": {
                        "name": tname,
                        "description": getattr(t, "description", "") or "",
                        "inputSchema": getattr(t, "inputSchema", None) or {},
                    },
                    "instruction": "",
                    "argOverrides": {},
                })

            structured = build_structured_tools_from_allowed_mcp_tools(
                allowed_tools=allowed,
                server_name=server_name,
                endpoint=endpoint,
                transport=transport,
                auth=auth,
                headers=headers,
            )
            all_tools.extend(structured)
            logger.info("Discovered %d tool(s) from '%s'", len(structured), name)

        self._missing_bearers = missing
        self._discovery_errors = errors
        return all_tools

    def _estimate_tools_tokens(self, tools: list) -> int:
        """Approx serialized token cost of the bound tool definitions. The
        FunctionDefinition OCI receives is dominated by the JSON schema of
        args, so we include it — under-counting makes trim fire too late."""
        estimate = 0
        for t in tools:
            name = getattr(t, "name", "") or ""
            desc = getattr(t, "description", "") or ""
            schema_json = ""
            schema = getattr(t, "args_schema", None)
            if schema is not None and hasattr(schema, "model_json_schema"):
                try:
                    schema_json = json.dumps(schema.model_json_schema())
                except Exception:
                    pass
            estimate += (len(name) + len(desc) + len(schema_json)) // 4 + 20
        return estimate

    async def _ensure_tools_loaded(self) -> None:
        if self._tools_loaded:
            return
        if self._load_lock is None:
            self._load_lock = asyncio.Lock()
        async with self._load_lock:
            if self._tools_loaded:
                return
            tools = await self._discover_and_build_tools()
            if not tools:
                return  # retry on a later request that carries a bearer
            self.agent = self._make_agent(tools)
            self._last_tool_names = [t.name for t in tools]
            self._tools_token_estimate = self._estimate_tools_tokens(tools)
            self._tools_loaded = True
            logger.info(
                "Bound %d tool(s): %s (≈%d tokens of tool overhead)",
                len(tools), self._last_tool_names, self._tools_token_estimate,
            )

    # ── Message-shape adapters ──────────────────────────────────
    # AIDP's checkpointer serializes messages to plain dicts. They come
    # back from aget_state() as dicts, not HumanMessage/AIMessage objects.
    # Every state-touching helper reads via these adapters.

    @staticmethod
    def _msg_get(msg, key, default=None):
        if isinstance(msg, dict):
            return msg.get(key, default)
        return getattr(msg, key, default)

    @classmethod
    def _msg_id(cls, msg):
        return cls._msg_get(msg, "id")

    @classmethod
    def _msg_tool_calls(cls, msg):
        tcs = cls._msg_get(msg, "tool_calls", None)
        if tcs is None:
            kwargs = cls._msg_get(msg, "additional_kwargs", None) or {}
            tcs = kwargs.get("tool_calls") if isinstance(kwargs, dict) else None
        return tcs or []

    @classmethod
    def _msg_tool_call_id(cls, msg):
        return cls._msg_get(msg, "tool_call_id")

    @classmethod
    def _msg_type(cls, msg):
        if isinstance(msg, dict):
            return msg.get("type") or msg.get("role") or "dict"
        return type(msg).__name__

    @classmethod
    def _final_reply(cls, result):
        """Reduce result['messages'] to the single user-facing answer.

        Prefer the LAST AIMessage with non-empty text content. This skips
        the trailing empty AIMessage / ToolMessage that some providers leave
        at the end of the run, which otherwise surfaces as an empty chat
        bubble. Falls back to the last message if no such AIMessage exists."""
        if not isinstance(result, dict):
            return result
        msgs = result.get("messages", []) or []
        if not msgs:
            return result

        def _text(m):
            c = cls._msg_get(m, "content", "")
            if isinstance(c, str):
                return c.strip()
            if isinstance(c, list):
                out = []
                for p in c:
                    if isinstance(p, str):
                        out.append(p)
                    elif isinstance(p, dict) and isinstance(p.get("text"), str):
                        out.append(p["text"])
                return " ".join(out).strip()
            return ""

        chosen = None
        for m in reversed(msgs):
            t = _text(m)
            # Require at least one alphanumeric char so punctuation-only junk
            # (e.g. a lone ".") is skipped in favour of a real prior answer.
            if cls._msg_type(m).lower() in ("ai", "aimessage") and any(c.isalnum() for c in t):
                chosen = m
                break
        if chosen is None:
            chosen = msgs[-1]
        return {**result, "messages": [chosen]}

    @staticmethod
    def _state_config(config):
        """Return a copy of `config` safe for state-reading calls.

        AIDP's pre_invoke_setup injects checkpoint_ns; a flat compiled graph
        has no subgraph by that name, so aget_state raises 'Subgraph ... not
        found'. Strip checkpoint_ns so state ops target the root namespace.
        ainvoke keeps the original config."""
        if not isinstance(config, dict):
            return config
        configurable = (config.get("configurable") or {}).copy()
        configurable.pop("checkpoint_ns", None)
        return {**config, "configurable": configurable}

    # ── Conversation state healing ──────────────────────────────
    async def _heal_orphan_tool_calls(self, config) -> int:
        """Remove AIMessage(tool_calls) that have no matching ToolMessage.
        Providers require the ToolMessage to IMMEDIATELY follow, so we delete
        the orphan AIMessage (via RemoveMessage) rather than append a fake."""
        if not self.agent or not config:
            return 0
        state_cfg = self._state_config(config)
        try:
            state = await self.agent.aget_state(state_cfg)
        except Exception as e:
            logger.debug("Could not fetch state for orphan check: %s", e)
            return 0

        messages = state.values.get("messages", []) if state and state.values else []
        if not messages:
            return 0

        expected, fulfilled = set(), set()
        for msg in messages:
            for tc in self._msg_tool_calls(msg):
                tcid = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                if tcid:
                    expected.add(tcid)
            tcid = self._msg_tool_call_id(msg)
            if tcid:
                fulfilled.add(tcid)

        orphans = expected - fulfilled
        if not orphans:
            return 0

        removals = []
        for msg in messages:
            msg_tc_ids = set()
            for tc in self._msg_tool_calls(msg):
                tcid = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                if tcid:
                    msg_tc_ids.add(tcid)
            if msg_tc_ids & orphans:
                msg_id = self._msg_id(msg)
                if msg_id:
                    removals.append(RemoveMessage(id=msg_id))

        if not removals:
            logger.warning(
                "Detected %d orphan tool_call(s) but messages have no .id to "
                "remove them by; cannot heal automatically", len(orphans),
            )
            return 0

        logger.warning(
            "Removing %d orphan AIMessage(s) (cleared %d unmatched tool_call ids)",
            len(removals), len(orphans),
        )
        await self.agent.aupdate_state(state_cfg, {"messages": removals})
        return len(removals)

    async def _hard_clear_history(self, config) -> int:
        """Remove EVERY message from conversation state, one RemoveMessage at
        a time (AIDP's checkpointer raises IndexError on large batched
        removals). Returns the count cleared."""
        if not self.agent or not config:
            return 0
        state_cfg = self._state_config(config)
        try:
            state = await self.agent.aget_state(state_cfg)
        except Exception as e:
            logger.debug("Could not fetch state for hard clear: %s", e)
            self._last_clear_strategy = f"state read failed: {e}"
            return 0

        messages = state.values.get("messages", []) if state and state.values else []
        if not messages:
            self._last_clear_strategy = "already empty"
            return 0

        removals = [RemoveMessage(id=self._msg_id(m)) for m in messages if self._msg_id(m)]
        if not removals:
            self._last_clear_strategy = "no removable ids"
            logger.warning(
                "Hard-clear: %d messages in state but NONE had an id field.",
                len(messages),
            )
            return 0

        succeeded, last_err = 0, ""
        for r in removals:
            try:
                await self.agent.aupdate_state(state_cfg, {"messages": [r]})
                succeeded += 1
            except Exception as e:
                last_err = f"{type(e).__name__}: {str(e)[:120]}"

        if succeeded:
            self._last_clear_strategy = f"one-at-a-time ({succeeded}/{len(removals)} removed)"
            logger.warning("Hard-clear: removed %d/%d msg(s)", succeeded, len(removals))
            return succeeded

        self._last_clear_strategy = f"failed (last error: {last_err})"
        logger.warning("Hard-clear: all %d removals failed: %s", len(removals), last_err or "n/a")
        return 0

    async def _trim_history_by_turns(self, config) -> int:
        """Hard cap by user-turn count. Cut is at a HumanMessage boundary so
        AIMessage(tool_calls)+ToolMessage pairs stay intact. No-op when
        MAX_TURNS_KEPT==0 or state has ≤ MAX_TURNS_KEPT turns."""
        if MAX_TURNS_KEPT <= 0 or not self.agent or not config:
            return 0
        state_cfg = self._state_config(config)
        try:
            state = await self.agent.aget_state(state_cfg)
        except Exception as e:
            logger.debug("Could not fetch state for turn trim: %s", e)
            return 0

        messages = state.values.get("messages", []) if state and state.values else []
        if not messages:
            return 0

        human_positions: list[int] = []
        for i in range(len(messages) - 1, -1, -1):
            if self._msg_type(messages[i]).lower() in ("human", "humanmessage"):
                human_positions.append(i)
                if len(human_positions) > MAX_TURNS_KEPT:
                    break
        if len(human_positions) <= MAX_TURNS_KEPT:
            return 0

        cut_idx = human_positions[MAX_TURNS_KEPT - 1]
        removals = [RemoveMessage(id=self._msg_id(m)) for m in messages[:cut_idx] if self._msg_id(m)]
        if not removals:
            return 0

        logger.info(
            "History turn-trim: dropped %d msg(s), kept %d/%d (MAX_TURNS_KEPT=%d)",
            len(removals), len(messages) - len(removals), len(messages), MAX_TURNS_KEPT,
        )
        await self.agent.aupdate_state(state_cfg, {"messages": removals})
        return len(removals)

    def _capture_usage_from_result(self, result) -> None:
        """Read OCI's usage.prompt_tokens from the latest AIMessage and derive
        the measured system+tools overhead (prompt_tokens − tokens(history we
        sent)). Lets _trim_history work from ground truth instead of the
        heuristic. Best-effort."""
        if _lc_count_tokens is None:
            return
        msgs = result.get("messages", []) if isinstance(result, dict) else []
        if not msgs:
            return

        prompt_tokens: int | None = None
        for msg in reversed(msgs):
            meta = (
                self._msg_get(msg, "response_metadata")
                or self._msg_get(msg, "additional_kwargs")
                or {}
            )
            if not isinstance(meta, dict):
                continue
            usage = meta.get("usage")
            if usage is None:
                continue
            if isinstance(usage, dict):
                pt = usage.get("prompt_tokens") or usage.get("promptTokens")
            else:
                pt = getattr(usage, "prompt_tokens", None) or getattr(usage, "promptTokens", None)
            if pt:
                prompt_tokens = int(pt)
                break
        if not prompt_tokens:
            return

        try:
            history_tokens = _lc_count_tokens(msgs[:-1]) if len(msgs) > 1 else 0
        except Exception as e:
            logger.debug("Could not count history tokens: %s", e)
            return

        measured_overhead = max(0, prompt_tokens - int(history_tokens))
        if LLM_MAX_CONTEXT_TOKENS > 0 and measured_overhead >= LLM_MAX_CONTEXT_TOKENS:
            logger.warning(
                "Derived overhead %d ≥ max_context_tokens %d — ignoring.",
                measured_overhead, LLM_MAX_CONTEXT_TOKENS,
            )
            return

        self._last_prompt_tokens = prompt_tokens
        self._measured_overhead_tokens = measured_overhead
        logger.info(
            "Captured usage: prompt_tokens=%d, history_tokens=%d, overhead=%d",
            prompt_tokens, history_tokens, measured_overhead,
        )

    async def _trim_history(self, config) -> int:
        """Drop oldest turns when running history would push the next LLM call
        past the token budget. Disabled when max_context_tokens==0. Keeps
        AIMessage(tool_calls)+ToolMessage pairs together via trim_messages."""
        if LLM_MAX_CONTEXT_TOKENS <= 0:
            return 0
        if _lc_trim_messages is None or _lc_count_tokens is None:
            logger.debug("trim_messages helper unavailable; skipping token trim")
            return 0
        if not self.agent or not config:
            return 0

        state_cfg = self._state_config(config)
        try:
            state = await self.agent.aget_state(state_cfg)
        except Exception as e:
            logger.debug("Could not fetch state for trim: %s", e)
            return 0

        messages = state.values.get("messages", []) if state and state.values else []
        if not messages:
            return 0

        has_human = any(
            self._msg_type(m).lower() in ("human", "humanmessage") for m in messages
        )
        if not has_human:
            logger.debug("Trim skipped: no HumanMessage in state (msgs=%d)", len(messages))
            return 0

        if self._measured_overhead_tokens is not None:
            overhead = (
                self._measured_overhead_tokens
                + LLM_RESPONSE_RESERVE_TOKENS
                + NEXT_HUMAN_SLACK_TOKENS
            )
            overhead_source = (
                f"measured(system+tools={self._measured_overhead_tokens},"
                f" reserve={LLM_RESPONSE_RESERVE_TOKENS}, new_msg≈{NEXT_HUMAN_SLACK_TOKENS})"
            )
        else:
            heuristic_prompt = len(AGENT_SYSTEM_PROMPT) // 4 + 1
            overhead = heuristic_prompt + self._tools_token_estimate + LLM_RESPONSE_RESERVE_TOKENS
            overhead_source = (
                f"heuristic(prompt≈{heuristic_prompt}, tools≈{self._tools_token_estimate},"
                f" reserve={LLM_RESPONSE_RESERVE_TOKENS})"
            )

        budget = LLM_MAX_CONTEXT_TOKENS - overhead
        if budget <= 0:
            logger.warning(
                "Trim disabled this turn: overhead %d ≥ max_context_tokens %d (source=%s).",
                overhead, LLM_MAX_CONTEXT_TOKENS, overhead_source,
            )
            self._trim_dropped_critical_context = True
            return 0

        try:
            kept = _lc_trim_messages(
                messages,
                max_tokens=budget,
                strategy="last",
                token_counter=_lc_count_tokens,
                allow_partial=False,
                start_on="human",
                include_system=False,
            )
        except Exception as e:
            logger.warning("trim_messages failed (skipping trim): %s", e)
            return 0

        if not kept:
            logger.warning(
                "Trim found no valid window within budget %d tok (source=%s, msgs=%d).",
                budget, overhead_source, len(messages),
            )
            self._trim_dropped_critical_context = True
            return 0

        kept_ids = {getattr(m, "id", None) for m in kept}
        kept_ids.discard(None)
        removals = [
            RemoveMessage(id=getattr(m, "id", None))
            for m in messages
            if getattr(m, "id", None) and getattr(m, "id", None) not in kept_ids
        ]
        if not removals:
            return 0

        logger.info(
            "History trim: dropped %d msg(s), kept %d/%d (budget %d tok, source=%s)",
            len(removals), len(kept), len(messages), budget, overhead_source,
        )
        await self.agent.aupdate_state(state_cfg, {"messages": removals})
        return len(removals)

    # ── Invoke (per user message) ───────────────────────────────
    async def invoke(
        self, user_query: str, **kwargs
    ) -> Union[Dict, AsyncGenerator[BaseMessage, None]]:
        # Step 1: setup-time errors
        if self._setup_error:
            return {"messages": [{"role": "ai", "content": self._setup_error}]}

        # Step 1.5: slash commands (bypass the LLM)
        if user_query and user_query.strip().lower() in ("/reset", "/clear", "/refresh", "/restart"):
            try:
                reset_config = pre_invoke_setup(**kwargs)
            except Exception:
                reset_config = {}
            try:
                cleared = await self._hard_clear_history(reset_config)
                strategy = self._last_clear_strategy or "unknown"
                if cleared > 0:
                    msg = (
                        f"🔄 Conversation history cleared ({cleared} message"
                        f"{'s' if cleared != 1 else ''} removed via {strategy}). "
                        "Tools and config remain loaded."
                    )
                else:
                    msg = (
                        f"🔄 Reset attempted but nothing was cleared "
                        f"(outcome: {strategy})."
                    )
                return {"messages": [{"role": "ai", "content": msg}]}
            except Exception as e:
                return {"messages": [{"role": "ai", "content":
                    f"Failed to clear history: {type(e).__name__}: {e}"}]}

        # Step 2: config
        config = pre_invoke_setup(**kwargs)

        if bool(kwargs.get("stream", False)):
            return self._stream_messages(user_query, config, kwargs)

        # Step 3: per-request session-variable context (resolves the bearer)
        token = pre_tool_setup(**kwargs)
        self._trim_dropped_critical_context = False
        try:
            # Step 4: lazy discovery
            await self._ensure_tools_loaded()
            if not self._tools_loaded:
                return {"messages": [{"role": "ai", "content": self._bearer_help_message()}]}

            # Step 5: heal orphan tool_calls from prior failures
            try:
                await self._heal_orphan_tool_calls(config)
            except Exception as e:
                logger.warning("Orphan healing failed (continuing): %s", e)

            # Step 6: trim history (turn cap and/or token budget)
            try:
                await self._trim_history_by_turns(config)
            except Exception as e:
                logger.warning("Turn-based trim failed (continuing): %s", e)
            try:
                await self._trim_history(config)
            except Exception as e:
                logger.warning("Token-based trim failed (continuing): %s", e)

            # Step 6.5: critical-context loss → clean reset + inform
            if self._trim_dropped_critical_context:
                logger.warning("Trim signalled critical-context loss — resetting")
                cleared = 0
                try:
                    cleared = await self._hard_clear_history(config)
                except Exception as clear_err:
                    logger.error("Hard clear failed after trim signal: %s", clear_err)
                return {"messages": [{"role": "ai", "content":
                    self._context_overflow_message(cleared)}]}

            # Step 7: run the agent
            message = {"messages": [dict(HumanMessage(content=user_query))]}
            try:
                result = await self.agent.ainvoke(input=message, config=config)
                self._capture_usage_from_result(result)
                return self._final_reply(result)
            except KeyError:
                # Bearer session variable missing at tool-call time.
                return {"messages": [{"role": "ai", "content": self._bearer_help_message()}]}
            except Exception as e:
                # Context overflow → clean reset + inform (no auto-retry; the
                # same query would overflow again).
                if _is_context_overflow_error(e):
                    logger.warning("Context-overflow mid-invoke (%s) — resetting", type(e).__name__)
                    cleared = 0
                    try:
                        cleared = await self._hard_clear_history(config)
                    except Exception as clear_err:
                        logger.error("Hard clear failed: %s", clear_err)
                    return {"messages": [{"role": "ai", "content":
                        self._context_overflow_message(cleared)}]}

                # INVALID_CHAT_HISTORY → hard-clear + retry once.
                if _is_invalid_history(e):
                    logger.warning("INVALID_CHAT_HISTORY — hard-clearing and retrying")
                    try:
                        await self._hard_clear_history(config)
                        result = await self.agent.ainvoke(input=message, config=config)
                        self._capture_usage_from_result(result)
                        return self._final_reply(result)
                    except Exception as e2:
                        logger.error("Hard-clear retry failed: %s", e2, exc_info=True)
                    return {"messages": [{"role": "ai", "content": self._try_again_message()}]}

                logger.exception("invoke error")
                return {"messages": [{"role": "ai", "content": self._try_again_message()}]}
        finally:
            post_tool_setup(token, kwargs=kwargs)

    async def _stream_messages(
        self, user_query: str, config, kwargs
    ) -> AsyncGenerator[BaseMessage, None]:
        token = pre_tool_setup(**kwargs)
        self._trim_dropped_critical_context = False
        try:
            await self._ensure_tools_loaded()
            if not self._tools_loaded:
                from langchain_core.messages import AIMessage
                yield AIMessage(content=self._bearer_help_message())
                return

            # Same context management as the non-stream path.
            try:
                await self._heal_orphan_tool_calls(config)
            except Exception as e:
                logger.warning("Orphan healing failed (continuing): %s", e)
            try:
                await self._trim_history_by_turns(config)
            except Exception as e:
                logger.warning("Turn-based trim failed (continuing): %s", e)
            try:
                await self._trim_history(config)
            except Exception as e:
                logger.warning("Token-based trim failed (continuing): %s", e)

            if self._trim_dropped_critical_context:
                from langchain_core.messages import AIMessage
                cleared = 0
                try:
                    cleared = await self._hard_clear_history(config)
                except Exception as clear_err:
                    logger.error("Hard clear failed after trim signal: %s", clear_err)
                yield AIMessage(content=self._context_overflow_message(cleared))
                return

            message = {"messages": [dict(HumanMessage(content=user_query))]}
            try:
                from aidputils.agents.auth.client.generative_ai_inference_v2_client import (  # noqa: F401
                    StreamingData,
                )
                stream = self.agent.astream(input=message, config=config, stream_mode="messages")
            except ImportError:
                stream = self.agent.astream(input=message, config=config)

            async for chunk in parse_stream_response(stream):
                yield chunk
        except Exception:
            logger.exception("Streaming error")
            raise
        finally:
            post_tool_setup(token, kwargs=kwargs)

    # ── Friendly messages ───────────────────────────────────────
    def _bearer_help_message(self) -> str:
        lines = [
            "I couldn't authenticate to the connected MCP server(s): no "
            "bearer token was available for the request.\n",
            "This agent does not generate tokens — the bearer must arrive "
            "as an AIDP session variable:\n",
        ]
        targets = self._missing_bearers or [
            (n, ic.get("bearer_session_variable")) for n, ic in _enabled_integrations()
        ]
        for name, var in targets:
            lines.append(f"  • {name}: session variable '{var}'")
        if self._discovery_errors:
            lines.append("\nOther discovery errors:")
            for name, err in self._discovery_errors:
                lines.append(f"  • {name}: {err[:200]}")
        lines.append(
            "\nDEV: set the value in the Agent Studio 'Variables' tab.\n"
            "PROD: the calling application must pass it on each request."
        )
        return "\n".join(lines)

    @staticmethod
    def _try_again_message() -> str:
        return (
            "I ran into a problem completing that request and couldn't "
            "recover automatically. Please try sending your message again. "
            "If it persists, type **/reset** to clear the conversation "
            "history and start fresh."
        )

    @staticmethod
    def _context_overflow_message(cleared: int) -> str:
        if cleared > 0:
            return (
                "The conversation grew larger than the configured context "
                f"budget could hold. I reset the history ({cleared} message"
                f"{'s' if cleared != 1 else ''} removed) to recover.\n\n"
                "All tools are still available — please re-send your question "
                "or ask a new one. For long sessions, narrower queries help "
                "(smaller LIMIT, aggregates instead of raw detail).\n\n"
                "You can also type **/reset** any time to start clean."
            )
        return (
            "The conversation grew larger than the configured context budget "
            "could hold, and I couldn't reset automatically. Please type "
            "**/reset** to start fresh."
        )


# ── Optional checkpointer (conversation memory) ─────────────────
# Defined after AgentBasic so a failure here can't shadow the class.
try:
    from aidputils.agents.toolkit.memory_helper import get_checkpoint_saver
    checkpointer = get_checkpoint_saver(AGENT_ID)
except Exception:
    checkpointer = globals().get("checkpointer", None)


# ─────────────────────────────────────────────────────────────────────
# CALLING THIS AGENT WITH SESSION VARIABLES (the bearer hand-off)
# ─────────────────────────────────────────────────────────────────────
# At runtime, invoke() receives session variables in **kwargs under the key
# `session_variables` (loaded by aidputils.pre_tool_setup into the per-request
# context the `{{sessionvariables...}}` placeholder resolves from, on every
# tool call). A value may be a plain string OR a {"value": ...} dict — the
# resolver normalizes both.
#
#   • DEV : set the value in the Agent Studio "Variables" tab — AIDP delivers
#           it to invoke() on every request.
#   • PROD: the calling application supplies it per request via the HTTP body
#           below; token generation stays OUTSIDE AIDP.
#
# ── HTTP call to a DEPLOYED agent ────────────────────────────────────
# ⚠️ Illustrative example shape (NOT yet verified end-to-end) — confirm
# against the AIDP API docs / your own deployment before relying on it.
# In this pattern, session variables travel in `metadata` as FLAT key→value
# pairs (the key is the full "sessionvariables.<name>"); the conversation is
# threaded by the `x-session-id` header; and auth is the OCI request signer.
# For THIS agent, pass each enabled integration's MCP bearer:
#
#   import requests
#   body = {
#       "isStreamEnabled": False,
#       "trace": False,
#       "input": [{
#           "role": "User",
#           "content": [{
#               "type": "INPUT_TEXT",
#               "text": "quantas faturas tem o cliente C001?",
#           }],
#       }],
#       "metadata": {
#           # One flat entry per ENABLED integration (string values):
#           "sessionvariables.cred.mcp.oac.bearer": "<the OAC MCP bearer>",
#           # "sessionvariables.cred.mcp.adb.bearer": "<the ADB MCP bearer>",
#           # "sessionvariables.cred.mcp.oic.bearer": "<the OIC MCP bearer>",
#       },
#   }
#   response = requests.post(
#       url=<insert-chat-url>,
#       params=None,
#       auth=<insert-oci-signer>,
#       json=body,
#       headers={"x-session-id": <insert-a-session-key>},
#   )
#
# Note: these bearers are declared isSystem/shouldLog=False (see
# `session_config`) so their values are never logged. If your tenancy locks
# a credential variable to the credential store, confirm whether callers may
# still override it via `metadata`.
# ─────────────────────────────────────────────────────────────────────


# ╔══════════════════════════════════════════════════════════════╗
# ║   Local smoke test (not used by AIDP)                        ║
# ╚══════════════════════════════════════════════════════════════╝
async def _main() -> None:
    agent = AgentBasic()
    agent.setup()
    fake_session_variables = {
        # "sessionvariables.cred.mcp.oac.bearer": {"value": "<PASTE_A_BEARER>"},
    }
    result = await agent.invoke(
        "ADD YOUR QUERY HERE",
        session_variables=fake_session_variables,
        thread_id="local-test-thread",
    )
    print("\nFinal Response:\n")
    print(result)


if __name__ == "__main__":
    asyncio.run(_main())
