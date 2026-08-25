"""
Multi-MCP Chat Agent — session-variable auth (no in-agent credentials).

This is the sibling of agent.py with ONE thing removed: all credential /
token MINTING. Everything else — MultiServerMCPClient + create_react_agent,
live tool discovery, and the full conversation context-management stack
(turn cap, token-budget trim, orphan-tool_call healing, /reset, context-
overflow + INVALID_CHAT_HISTORY recovery) — is kept exactly as in agent.py.

────────────────────────────────────────────────────────────────────────
WHAT CHANGED vs agent.py
────────────────────────────────────────────────────────────────────────
agent.py mints the bearer for each MCP server itself (ADB password grant,
OAC JWT assertion, OIC client_credentials) and holds the secrets. That is
NOT the product-intended model.

Here the agent NEVER mints a token. The bearer for each MCP server is read
from the per-request AIDP *session variables* (the same source the
session-variables-agent uses) and injected into the MultiServerMCPClient
headers:

  • DEV : store a hand-made bearer in the Agent Studio "Variables" tab. It
          arrives on each request as a session variable.
  • PROD: the calling application passes the bearer per request. Token
          generation stays OUTSIDE AIDP.

How the bearer reaches this code: aidputils.pre_tool_setup(**kwargs) loads
kwargs["session_variables"] into chat_context.session_context_var; we read
the configured variable for each integration from there (and from kwargs as
a fallback), normalize the value (plain string OR {"value": ...} dict), and
build the MCP client headers with it.

REMOVED vs agent.py: _adb_bearer_token, _oic_bearer_token, the whole OAC
JWT machinery (_mint_jwt, _fetch_oac_access_token, key staging, token TTL),
the proactive _ensure_fresh_tokens refresh, and the reactive token-REMINT
retry. (Connection-level retry is kept; an auth 401 now means "the caller
must supply a valid bearer", not "re-mint and retry".)

────────────────────────────────────────────────────────────────────────
TRADE-OFF you should know (kept honest)
────────────────────────────────────────────────────────────────────────
MultiServerMCPClient bakes the Authorization header into the client at
build time. To honour a per-user bearer we rebuild the MCP client whenever
the resolved bearer set changes (cheap when it doesn't). This is correct
for sequential / single-tenant use. Under TRULY concurrent users with
DIFFERENT bearers it can race (the shared graph uses whichever bearer was
last built). If you need strict per-call, per-user bearer isolation under
concurrency, use the session-variables-agent variant — its aidputils MCP
tools resolve the `{{sessionvariables...}}` placeholder per call.

Config: ./config.yaml. This agent reads the ENDPOINT fields only — for ADB
`ocid`+`region`, for OAC `url`, for OIC `mcp_url` — plus an optional
`bearer_session_variable` per integration (defaults below). It ignores the
credential fields agent.py uses (user/password/client_secret/private key…).
"""

import asyncio
import hashlib
import json
import logging
import os
import time
import traceback
from pathlib import Path

from langchain_core.messages import HumanMessage, RemoveMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

# Conversation-history trimming helpers. Available in langchain_core ≥0.3.
# Fall back to None if the AIDP runtime ships an older version.
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
)
from aidputils.agents.toolkit.configs import OCIAIConf

logger = logging.getLogger(__name__)


# ╔══════════════════════════════════════════════════════════════╗
# ║   MAX_TURNS_KEPT — hard cap on conversation history          ║
# ║   (see agent.py for the full rationale). Set 0 to disable.    ║
# ╚══════════════════════════════════════════════════════════════╝
MAX_TURNS_KEPT = 5

# Budget headroom reserved for the upcoming HumanMessage.
NEXT_HUMAN_SLACK_TOKENS = 200


# ╔══════════════════════════════════════════════════════════════╗
# ║   System-prompt strings (ported from agent.py)               ║
# ╚══════════════════════════════════════════════════════════════╝
SYSTEM_PROMPT_INTRO = (
    "You are a helpful data assistant with access to tools loaded "
    "from one or more MCP servers. Users ask data questions in "
    "plain English; you answer them by calling the right tool."
)
ADB_TOOL_DESCRIPTION = (
    "  - ADB SelectAI tool: a natural-language-to-SQL agent over "
    "an Oracle Autonomous Database. Pass questions in natural "
    "language, fully qualified with table / column names whenever "
    "the user has named them."
)
OAC_TOOL_DESCRIPTION = (
    "  - Oracle Analytics Cloud (OAC) tools: query OAC subject "
    "areas / datasets with governance-aware Logical SQL. Use "
    "these for analytical / reporting questions about pre-modeled "
    "data."
)
OIC_TOOL_DESCRIPTION = (
    "  - Oracle Integration Cloud (OIC) tools: invoke OIC "
    "integrations exposed by the OIC project's MCP server. Use "
    "these for questions about integrations, audit reports, and "
    "orchestration exposed by the OIC instance."
)
RULE_RETRY_ON_BAD_RESULT = (
    "  - If a previous tool call returned wrong, empty, or "
    "unexpected results, do NOT stop using tools. User "
    "dissatisfaction is a signal to make a SMARTER call (different "
    "table, clearer filter, discovery step first) — never to give "
    "up. Re-issue the tool with corrected parameters."
)
RULE_DISCOVER_FIRST = (
    "  - Pick the right tool for each question. If you're unsure "
    "which table or column to use, run a discovery call FIRST "
    "(e.g. list tables, describe columns) and base the real query "
    "on the result. Do NOT guess."
)
RULE_ACT_DONT_ANNOUNCE = (
    "  - ACT, don't announce. NEVER say you 'will' call a tool, "
    "'will query now', 'let me check', 'vou consultar agora', etc. "
    "and then stop. If you need data, call the tool IMMEDIATELY in "
    "this same turn, then answer. Every turn must end with EITHER a "
    "tool call OR a complete answer — never a promise to act later."
)
RULE_ALWAYS_SHOW_DATA = (
    "  - ALWAYS include the data. After a tool returns rows, your "
    "reply MUST contain the actual values (a compact Markdown table "
    "or list). NEVER say 'here are the results' / 'apareceu?' "
    "without the rows, and NEVER reply with empty content."
)
RULE_ORACLE_SQL = (
    "  - Oracle SQL dialect. Every connected system is an Oracle "
    "product, so any SQL you write or ask a tool to run MUST be "
    "Oracle-compatible: FETCH FIRST n ROWS ONLY (never LIMIT/TOP), "
    "SYSDATE/CURRENT_DATE, NVL or COALESCE, TO_DATE/TO_CHAR for "
    "dates, concatenation with || . For OAC use its Logical SQL. "
    "Never emit MySQL/PostgreSQL/SQL Server-specific syntax."
)
RULE_PRESERVE_INTENT = (
    "  - Preserve the user's intent. You MAY enrich the question "
    "with fully-qualified table names or filter clarifications when "
    "the user has specified them — but never invent filter values, "
    "and never substitute the user's intent with your own."
)
RULE_FOLLOWUPS = (
    "  - For follow-ups, rewrite using prior turn context. MCP "
    "tools see only the prompt you pass them — they have no "
    "awareness of earlier conversation turns."
)
RULE_SUMMARIZE = (
    "  - After a successful tool call, summarize in plain English. "
    "Don't dump raw JSON or row arrays unless the user asks for raw "
    "data. For numeric answers, state the number AND the table it "
    "came from."
)
RULE_NO_INVENTING = (
    "  - Never invent numbers. If a tool returns an ERROR (e.g. 502, "
    "timeout, empty / no result) or 0 rows, report the failure "
    "plainly — name the tool/table and STOP. NEVER fabricate, "
    "estimate, or extrapolate from a failed call or a partial sample "
    "(do NOT say 'N based on a sample of N'). A failed call is not data."
)
ADB_RULE_ACTIONS = (
    "  - For SelectAI: action='runsql' to execute and return data, "
    "'showsql' to see generated SQL without running, 'explainsql' "
    "to explain the SQL. If you suspect SelectAI may be routing to "
    "the wrong table, use 'showsql' first to verify."
)
ADB_RULE_QUALIFY_TABLES = (
    "  - When the user names a specific ADB table, every SelectAI "
    "prompt you generate MUST include the fully-qualified name "
    "verbatim. Do not let SelectAI pick the table when the user has "
    "already named one. If SelectAI still routes to a different "
    "table, call it out to the user and re-issue with the table name "
    "embedded more forcefully."
)
OAC_RULE_DISCOVERY = (
    "  - For OAC Logical SQL: use discover_data and describe_data "
    "BEFORE execute_logical_sql so you know exact table / column "
    "names."
)


# ╔══════════════════════════════════════════════════════════════╗
# ║   Config loading (ported from agent.py)                      ║
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
                _CFG_PATHS_TRIED.append(f"(rglob match) {p}")
                return p
        except Exception:
            pass
    return None


try:
    import yaml  # PyYAML — confirmed pre-installed in AIDP runtime
    found = _find_config_yaml()
    if found is None:
        raise FileNotFoundError(
            "config.yaml not found in any expected location. Searched:\n"
            + "\n".join(f"  • {p}" for p in _CFG_PATHS_TRIED)
            + "\n\nUpload config.yaml next to this file in the AIDP project."
        )
    CFG = yaml.safe_load(found.read_text()) or {}
    logger.info("Loaded config.yaml from %s", found)
except Exception:
    import traceback as _tb
    _CFG_INIT_ERROR = _tb.format_exc()

LLM_COMPARTMENT_ID = CFG.get("llm", {}).get("compartment_id", "")
LLM_REGION         = CFG.get("llm", {}).get("region", "us-ashburn-1")
LLM_MODEL_ID       = CFG.get("llm", {}).get("model_id", "")
LLM_MODEL_ARGS     = CFG.get("llm", {}).get("model_args", {}) or {}
LLM_MAX_CONTEXT_TOKENS      = int(CFG.get("llm", {}).get("max_context_tokens", 0) or 0)
LLM_RESPONSE_RESERVE_TOKENS = int(CFG.get("llm", {}).get("response_reserve_tokens", 1024) or 1024)

ADB_CFG     = CFG.get("integrations", {}).get("adb", {}) or {}
ADB_ENABLED = bool(ADB_CFG.get("enabled", False))
OAC_CFG     = CFG.get("integrations", {}).get("oac", {}) or {}
OAC_ENABLED = bool(OAC_CFG.get("enabled", False))
OIC_CFG     = CFG.get("integrations", {}).get("oic", {}) or {}
OIC_ENABLED = bool(OIC_CFG.get("enabled", False))

# Bearer session-variable names (override per integration in config.yaml).
ADB_BEARER_VAR = ADB_CFG.get("bearer_session_variable", "sessionvariables.cred.mcp.adb.bearer")
OAC_BEARER_VAR = OAC_CFG.get("bearer_session_variable", "sessionvariables.cred.mcp.oac.bearer")
OIC_BEARER_VAR = OIC_CFG.get("bearer_session_variable", "sessionvariables.cred.mcp.oic.bearer")

_INTEGRATION_ENABLED = {"adb": ADB_ENABLED, "oac": OAC_ENABLED, "oic": OIC_ENABLED}
_INTEGRATION_BEARER_VARS = {"adb": ADB_BEARER_VAR, "oac": OAC_BEARER_VAR, "oic": OIC_BEARER_VAR}


def _build_system_prompt() -> str:
    parts: list[str] = [SYSTEM_PROMPT_INTRO]
    if ADB_ENABLED:
        parts.append(ADB_TOOL_DESCRIPTION)
    if OAC_ENABLED:
        parts.append(OAC_TOOL_DESCRIPTION)
    if OIC_ENABLED:
        parts.append(OIC_TOOL_DESCRIPTION)

    parts.append("")
    parts.append("Rules:")
    parts.append(RULE_ACT_DONT_ANNOUNCE)
    parts.append(RULE_ALWAYS_SHOW_DATA)
    parts.append(RULE_RETRY_ON_BAD_RESULT)
    parts.append(RULE_DISCOVER_FIRST)
    if ADB_ENABLED:
        parts.append(ADB_RULE_ACTIONS)
        parts.append(ADB_RULE_QUALIFY_TABLES)
    if OAC_ENABLED:
        parts.append(OAC_RULE_DISCOVERY)
    parts.append(RULE_ORACLE_SQL)
    parts.append(RULE_PRESERVE_INTENT)
    parts.append(RULE_FOLLOWUPS)
    parts.append(RULE_SUMMARIZE)
    parts.append(RULE_NO_INVENTING)
    return "\n".join(parts)


AGENT_SYSTEM_PROMPT = _build_system_prompt()

llm_conf = OCIAIConf(
    model_provider="generic",
    compartment_id=LLM_COMPARTMENT_ID,
    model_args=LLM_MODEL_ARGS,
    endpoint=f"https://inference.generativeai.{LLM_REGION}.oci.oraclecloud.com",
    model_id=LLM_MODEL_ID,
    guardrails_config={},
)

checkpointer = globals().get("checkpointer", None)


# ╔══════════════════════════════════════════════════════════════╗
# ║   MCP endpoint builders (no credentials needed)              ║
# ╚══════════════════════════════════════════════════════════════╝
def _adb_mcp_url() -> str:
    return (
        f"https://dataaccess.adb.{ADB_CFG.get('region', '')}.oraclecloudapps.com"
        f"/adb/mcp/v1/databases/{ADB_CFG.get('ocid', '')}"
    )


def _oac_mcp_url() -> str:
    return f"{str(OAC_CFG.get('url', '')).rstrip('/')}/api/mcp"


def _oic_mcp_url() -> str:
    return OIC_CFG.get("mcp_url", "")


# ╔══════════════════════════════════════════════════════════════╗
# ║   Session-variable bearer resolution (from request context)  ║
# ╚══════════════════════════════════════════════════════════════╝
def _extract_session_variables(kwargs: dict) -> dict:
    """Best-effort collection of the per-request session-variable map.

    Primary source is chat_context.session_context_var (populated by
    pre_tool_setup from kwargs["session_variables"]); we also merge any
    dict found directly under kwargs for robustness."""
    found: dict = {}
    try:
        from aidputils.agents.toolkit import chat_context
        ctx = chat_context.session_context_var.get()
        if isinstance(ctx, dict):
            found.update(ctx)
    except Exception:
        pass
    for key in ("session_variables", "variables"):
        v = kwargs.get(key)
        if isinstance(v, dict):
            found.update(v)
    return found


def _normalize_sv_value(entry) -> str | None:
    """Normalize a session-variable entry to a token string. Accepts a
    {"value": ...} dict (SessionVariableDetails) or a plain scalar."""
    if entry is None:
        return None
    if isinstance(entry, dict):
        val = entry.get("value")
        return str(val) if val not in (None, "") else None
    if entry == "":
        return None
    return str(entry)


def _resolve_bearer(var_name: str | None, sv: dict) -> str | None:
    if not var_name or not isinstance(sv, dict):
        return None
    candidates = [var_name]
    if var_name.startswith("sessionvariables."):
        candidates.append(var_name[len("sessionvariables."):])
    for k in candidates:
        if k in sv:
            val = _normalize_sv_value(sv[k])
            if val:
                return val
    return None


def _build_mcp_config(bearers: dict) -> dict:
    """Build the MultiServerMCPClient config for enabled integrations that
    have a resolved bearer. Bearers are baked into the headers here."""
    cfg: dict = {}
    if ADB_ENABLED and bearers.get("adb"):
        cfg["adb_selectai"] = {
            "transport": "streamable_http",
            "url": _adb_mcp_url(),
            "headers": {"Authorization": f"Bearer {bearers['adb']}"},
        }
    if OAC_ENABLED and bearers.get("oac"):
        cfg["oac_analytics"] = {
            "transport": "streamable_http",
            "url": _oac_mcp_url(),
            "headers": {"Authorization": f"Bearer {bearers['oac']}"},
        }
    if OIC_ENABLED and bearers.get("oic"):
        cfg["oic_integrations"] = {
            "transport": "streamable_http",
            "url": _oic_mcp_url(),
            "headers": {
                "Authorization": f"Bearer {bearers['oic']}",
                "Accept": "application/json, text/event-stream",
            },
        }
    return cfg


# ╔══════════════════════════════════════════════════════════════╗
# ║   Error classifiers (ported from agent.py)                   ║
# ╚══════════════════════════════════════════════════════════════╝
def _is_auth_error(e: Exception) -> bool:
    msg = str(e).lower()
    return any(s in msg for s in ("401", "403", "unauthorized", "forbidden", "expired"))


def _is_invalid_history(e: BaseException) -> bool:
    sub = getattr(e, "exceptions", None)
    if sub:
        return any(_is_invalid_history(s) for s in sub)
    msg = str(e)
    return (
        "INVALID_CHAT_HISTORY" in msg
        or "tool_calls that do not have a corresponding ToolMessage" in msg
    )


def _is_context_overflow_error(e: BaseException) -> bool:
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


def _is_connection_error(e: BaseException) -> bool:
    """Connection-level (NOT auth) failure worth one rebuild+retry."""
    sub = getattr(e, "exceptions", None)
    if sub:
        return any(_is_connection_error(s) for s in sub)
    if isinstance(e, Exception) and _is_auth_error(e):
        return False
    cls_name = type(e).__name__
    if cls_name in ("ClosedResourceError", "BrokenResourceError",
                    "ConnectionError", "RemoteProtocolError"):
        return True
    msg = str(e).lower()
    return any(s in msg for s in (
        "closedresourceerror", "brokenresourceerror",
        "connection closed", "connection reset", "remote disconnected",
    ))


# ╔══════════════════════════════════════════════════════════════╗
# ║   AgentBasic — what AIDP discovers                           ║
# ╚══════════════════════════════════════════════════════════════╝
class AgentBasic:
    def __init__(self) -> None:
        self.llm = None
        self.graph = None

        self._tools_loaded = False
        self._self_stage_error: str | None = None
        self._load_lock = None

        # Bearer set the current MCP client was built with — rebuild when it
        # changes (per-user bearer support; see module docstring trade-off).
        self._loaded_bearer_sig: str | None = None
        self._missing_bearers: list[tuple[str, str]] = []

        self._last_tool_names: list[str] = []
        self._tools_token_estimate: int = 0

        self._last_prompt_tokens: int | None = None
        self._measured_overhead_tokens: int | None = None
        self._trim_dropped_critical_context: bool = False
        self._last_clear_strategy: str = ""

    # ── Setup (sync) ────────────────────────────────────────────
    def setup(self) -> None:
        logger.info(
            "Initializing Multi-MCP Chat Agent (session-variable auth) — "
            "enabled: ADB=%s, OAC=%s, OIC=%s",
            ADB_ENABLED, OAC_ENABLED, OIC_ENABLED,
        )
        if _CFG_INIT_ERROR:
            self._self_stage_error = (
                "config.yaml could not be loaded.\n\n"
                f"Error:\n{_CFG_INIT_ERROR}\n\n"
                "Make sure config.yaml is uploaded next to this file."
            )
            return
        if not (ADB_ENABLED or OAC_ENABLED or OIC_ENABLED):
            self._self_stage_error = (
                "No integration enabled in config.yaml. Set 'enabled: true' "
                "under integrations.adb / .oac / .oic."
            )
            return

        self.llm = init_oci_llm(llm_conf)
        self.graph = create_react_agent(
            self.llm, [], prompt=AGENT_SYSTEM_PROMPT, checkpointer=checkpointer,
        )
        logger.info("Stub graph ready; MCP tools load lazily on first invoke")

    # ── Bearer resolution ───────────────────────────────────────
    def _resolve_bearers(self, kwargs: dict) -> dict:
        sv = _extract_session_variables(kwargs)
        bearers: dict = {}
        missing: list[tuple[str, str]] = []
        for name in ("adb", "oac", "oic"):
            if not _INTEGRATION_ENABLED[name]:
                continue
            var = _INTEGRATION_BEARER_VARS[name]
            b = _resolve_bearer(var, sv)
            if b:
                bearers[name] = b
            else:
                missing.append((name, var))
        self._missing_bearers = missing
        return bearers

    @staticmethod
    def _bearer_signature(bearers: dict) -> str:
        h = hashlib.sha256()
        for k in sorted(bearers):
            h.update(k.encode())
            h.update(b"=")
            h.update(bearers[k].encode())
            h.update(b";")
        return h.hexdigest()

    # ── MCP loading (ephemeral sessions per tool call) ─────────
    async def _load_all_mcp_tools(self, bearers: dict) -> None:
        cfg = _build_mcp_config(bearers)
        if not cfg:
            self._last_tool_names = []
            return
        client = MultiServerMCPClient(cfg)
        tools = await client.get_tools()

        self._last_tool_names = [t.name for t in tools]

        estimate = 0
        for t in tools:
            name = getattr(t, "name", "") or ""
            desc = getattr(t, "description", "") or ""
            schema_json = ""
            schema = getattr(t, "args_schema", None)
            if schema is not None:
                if hasattr(schema, "model_json_schema"):
                    try:
                        schema_json = json.dumps(schema.model_json_schema())
                    except Exception:
                        pass
                elif hasattr(schema, "schema"):
                    try:
                        schema_json = json.dumps(schema.schema())
                    except Exception:
                        pass
            estimate += (len(name) + len(desc) + len(schema_json)) // 4 + 20
        self._tools_token_estimate = estimate

        logger.info(
            "Loaded %d tool(s) from %d MCP server(s) [ephemeral sessions]: %s "
            "(≈%d tokens of tool overhead)",
            len(tools), len(cfg), self._last_tool_names, self._tools_token_estimate,
        )
        self.graph = create_react_agent(
            self.llm, tools, prompt=AGENT_SYSTEM_PROMPT, checkpointer=checkpointer,
        )
        self._tools_loaded = True

    async def _ensure_mcp_for_bearers(self, bearers: dict) -> None:
        """(Re)build MCP when not yet loaded or when the bearer set changed."""
        sig = self._bearer_signature(bearers)
        if self._tools_loaded and sig == self._loaded_bearer_sig:
            return
        async with self._load_lock:
            if self._tools_loaded and sig == self._loaded_bearer_sig:
                return
            await self._load_all_mcp_tools(bearers)
            self._loaded_bearer_sig = sig

    # ── Message-shape adapters (ported from agent.py) ───────────
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
            t = msg.get("type") or msg.get("role")
            return t or "dict"
        return type(msg).__name__

    @classmethod
    def _msg_content(cls, msg):
        c = cls._msg_get(msg, "content", "")
        return c if isinstance(c, str) else ""

    @staticmethod
    def _state_config(config):
        if not isinstance(config, dict):
            return config
        configurable = (config.get("configurable") or {}).copy()
        configurable.pop("checkpoint_ns", None)
        return {**config, "configurable": configurable}

    # ── Conversation state healing (ported from agent.py) ───────
    async def _heal_orphan_tool_calls(self, config) -> int:
        if not self.graph or not config:
            return 0
        state_cfg = self._state_config(config)
        try:
            state = await self.graph.aget_state(state_cfg)
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
            "Removing %d orphan AIMessage(s) (cleared %d unmatched tool_call ids: %s)",
            len(removals), len(orphans), list(orphans),
        )
        await self.graph.aupdate_state(state_cfg, {"messages": removals})
        return len(removals)

    async def _hard_clear_history(self, config) -> int:
        if not self.graph or not config:
            return 0
        state_cfg = self._state_config(config)
        try:
            state = await self.graph.aget_state(state_cfg)
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
                await self.graph.aupdate_state(state_cfg, {"messages": [r]})
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

    async def _clear_last_turn(self, config) -> int:
        if not self.graph or not config:
            return 0
        state_cfg = self._state_config(config)
        try:
            state = await self.graph.aget_state(state_cfg)
        except Exception as e:
            logger.debug("Could not fetch state for last-turn clear: %s", e)
            return 0

        messages = state.values.get("messages", []) if state and state.values else []
        if not messages:
            return 0

        cut_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if self._msg_type(messages[i]).lower() in ("human", "humanmessage"):
                cut_idx = i
                break
        if cut_idx is None:
            return 0

        removals = [RemoveMessage(id=self._msg_id(m)) for m in messages[cut_idx:] if self._msg_id(m)]
        if not removals:
            return 0

        succeeded = 0
        for r in removals:
            try:
                await self.graph.aupdate_state(state_cfg, {"messages": [r]})
                succeeded += 1
            except Exception as e:
                logger.debug("Last-turn remove failed for one msg: %s", e)
        logger.warning("Last-turn clear: removed %d/%d msg(s)", succeeded, len(removals))
        return succeeded

    async def _trim_history_by_turns(self, config) -> int:
        if MAX_TURNS_KEPT <= 0 or not self.graph or not config:
            return 0
        state_cfg = self._state_config(config)
        try:
            state = await self.graph.aget_state(state_cfg)
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
        await self.graph.aupdate_state(state_cfg, {"messages": removals})
        return len(removals)

    def _capture_usage_from_result(self, result) -> None:
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
        if LLM_MAX_CONTEXT_TOKENS <= 0:
            return 0
        if _lc_trim_messages is None or _lc_count_tokens is None:
            logger.debug("trim_messages helper unavailable; skipping token trim")
            return 0
        if not self.graph or not config:
            return 0

        state_cfg = self._state_config(config)
        try:
            state = await self.graph.aget_state(state_cfg)
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
        await self.graph.aupdate_state(state_cfg, {"messages": removals})
        return len(removals)

    # ── Diagnostics & friendly messages ─────────────────────────
    def _runtime_diagnostics(self) -> str:
        lines = [
            f"Path.home() = {Path.home()}",
            f"euid/uid    = {os.geteuid()}/{os.getuid()}",
            f"USER env    = {os.environ.get('USER', '<unset>')}",
            f"cwd         = {Path.cwd()}",
            f"clock epoch = {int(time.time())}",
            "",
            f"enabled     : ADB={ADB_ENABLED}  OAC={OAC_ENABLED}  OIC={OIC_ENABLED}",
            f"tools_loaded: {self._tools_loaded} ({self._last_tool_names})",
            f"missing bearers: {[n for n, _ in self._missing_bearers]}",
        ]
        return "\n".join(lines)

    def _bearer_help_message(self) -> str:
        lines = [
            "I couldn't authenticate to the connected MCP server(s): no bearer "
            "token was available for the request.\n",
            "This agent does not generate tokens — the bearer must arrive as an "
            "AIDP session variable:\n",
        ]
        targets = self._missing_bearers or [
            (n, _INTEGRATION_BEARER_VARS[n]) for n in ("adb", "oac", "oic")
            if _INTEGRATION_ENABLED[n]
        ]
        for name, var in targets:
            lines.append(f"  • {name}: session variable '{var}'")
        lines.append(
            "\nDEV: set the value in the Agent Studio 'Variables' tab.\n"
            "PROD: the calling application must pass it on each request."
        )
        return "\n".join(lines)

    @staticmethod
    def _try_again_message() -> str:
        return (
            "I ran into a problem completing that request and couldn't recover "
            "automatically. Please try sending your message again. If it "
            "persists, type **/reset** to clear the conversation history."
        )

    @staticmethod
    def _context_overflow_message(cleared: int) -> str:
        if cleared > 0:
            return (
                "The conversation grew larger than the configured context budget "
                f"could hold. I reset the history ({cleared} message"
                f"{'s' if cleared != 1 else ''} removed) to recover.\n\n"
                "All tools are still available — please re-send your question. "
                "For long sessions, narrower queries help (smaller row caps, "
                "aggregates instead of raw detail).\n\n"
                "You can also type **/reset** any time to start clean."
            )
        return (
            "The conversation grew larger than the configured context budget "
            "could hold, and I couldn't reset automatically. Please type "
            "**/reset** to start fresh."
        )

    # ── Invoke (per user message) ───────────────────────────────
    async def invoke(self, user_query: str, **kwargs):
        # Step 1: setup-time errors
        if self._self_stage_error:
            return {"messages": [{"role": "ai", "content":
                f"Self-stage failed at setup:\n{self._self_stage_error}\n\n"
                f"Runtime diagnostics:\n{self._runtime_diagnostics()}"
            }]}

        # Step 1.5: slash commands
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
                        "Auth, tools, and config remain loaded."
                    )
                else:
                    msg = f"🔄 Reset attempted but nothing was cleared (outcome: {strategy})."
                return {"messages": [{"role": "ai", "content": msg}]}
            except Exception as e:
                return {"messages": [{"role": "ai", "content":
                    f"Failed to clear history: {type(e).__name__}: {e}"}]}

        # Step 2: OCI Gen AI auth context
        try:
            config = pre_invoke_setup(**kwargs)
        except Exception as e:
            logger.warning("pre_invoke_setup failed, using empty config: %s", e)
            config = {}

        if self._load_lock is None:
            self._load_lock = asyncio.Lock()

        # Step 3: load the per-request session-variable context so the bearer
        # is resolvable, then resolve it.
        token = pre_tool_setup(**kwargs)
        self._trim_dropped_critical_context = False
        try:
            bearers = self._resolve_bearers(kwargs)
            if not bearers:
                return {"messages": [{"role": "ai", "content": self._bearer_help_message()}]}

            # Step 4: (re)build MCP if needed (first load or bearer changed)
            try:
                await self._ensure_mcp_for_bearers(bearers)
            except Exception as e:
                logger.error("MCP load failed: %s", e, exc_info=True)
                if _is_auth_error(e):
                    return {"messages": [{"role": "ai", "content": self._bearer_help_message()}]}
                return {"messages": [{"role": "ai", "content":
                    f"MCP setup error: {type(e).__name__}: {e}\n\n"
                    f"Runtime diagnostics:\n{self._runtime_diagnostics()}\n\n"
                    f"Stack trace (last 800 chars):\n{traceback.format_exc()[-800:]}"
                }]}

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

            # Step 7: run the react-agent graph
            messages = {"messages": [dict(HumanMessage(content=user_query))]}
            try:
                result = await self.graph.ainvoke(messages, config=config)
                self._capture_usage_from_result(result)
                return result
            except Exception as e:
                # Connection-level error → rebuild MCP (same bearer) + retry once.
                if _is_connection_error(e):
                    logger.warning("Connection error mid-invoke (%s); rebuilding MCP and retrying", type(e).__name__)
                    try:
                        async with self._load_lock:
                            await self._load_all_mcp_tools(bearers)
                            self._loaded_bearer_sig = self._bearer_signature(bearers)
                        try:
                            await self._heal_orphan_tool_calls(config)
                        except Exception as heal_err:
                            logger.warning("Orphan healing on retry failed: %s", heal_err)
                        result = await self.graph.ainvoke(messages, config=config)
                        self._capture_usage_from_result(result)
                        return result
                    except Exception as e2:
                        logger.error("Retry after rebuild failed: %s", e2, exc_info=True)
                        return {"messages": [{"role": "ai", "content": self._try_again_message()}]}

                # Auth error → the caller's bearer is missing/invalid/expired.
                if _is_auth_error(e):
                    logger.warning("Auth error mid-invoke (%s)", type(e).__name__)
                    return {"messages": [{"role": "ai", "content": self._bearer_help_message()}]}

                # Context overflow → clean reset + inform (no auto-retry).
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
                        result = await self.graph.ainvoke(messages, config=config)
                        self._capture_usage_from_result(result)
                        return result
                    except Exception as e2:
                        logger.error("Hard-clear retry failed: %s", e2, exc_info=True)
                    return {"messages": [{"role": "ai", "content": self._try_again_message()}]}

                logger.error("invoke error: %s", e, exc_info=True)
                return {"messages": [{"role": "ai", "content": self._try_again_message()}]}
        finally:
            post_tool_setup(token, kwargs=kwargs)
