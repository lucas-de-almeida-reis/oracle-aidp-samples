"""
Multi-MCP Chat Agent for AIDP — Oracle ADB SelectAI + OAC + OIC.

This is the entry point AIDP looks for when it deploys this folder as
an agent. AIDP loads this file, finds the AgentBasic class, and calls
its setup() once and invoke() per chat message.

Reads configuration from ./config.yaml (the sibling file). Each
integration (adb / oac / oic) can be turned on or off independently.
See README.md for the end-to-end setup walkthrough.

Quick architecture:
  • Agent state lives in memory only (one Python process serves all
    chat users from the same AgentBasic instance).
  • MCP tool execution happens via short-lived HTTPS requests — there
    are no persistent network connections to keep alive.
  • Auth is on-demand only. All three integrations authenticate via
    OAuth 2.0 grants — none use a rotating refresh chain:
        - ADB:  password grant (RFC 6749 §4.3)
        - OAC:  JWT Bearer / User Assertion (RFC 7523), local-signed
                from a long-lived private key
        - OIC:  client_credentials (RFC 6749 §4.4)
    So we don't run any background refresh loop.
  • OAC's access_token has a 5-minute TTL — short enough that AIDP's
    idle-suspend can wake the agent up with a long-expired bearer
    cached in MCP's headers. To handle that, every invoke() runs
    _ensure_fresh_oac_token() at the top, which checks the cached
    JWT's exp claim and mints a fresh one if there's <60s remaining.
    Cheap when fresh (one JWT decode); ~300ms refresh when needed.
  • If a tool call still fails with 401/connection error despite the
    proactive refresh, the reactive retry in invoke() catches that,
    rebuilds MCP, heals any orphan tool_calls, and retries once.
"""

import asyncio
import base64
import json
import logging
import os
import time
import traceback
import uuid
from pathlib import Path

import requests

from langchain_core.messages import HumanMessage, RemoveMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

# Conversation-history trimming helpers. Available in langchain_core ≥0.3.
# Fall back to None if the AIDP runtime ships an older version — the trim
# feature then no-ops gracefully (logs a debug line and skips).
try:
    from langchain_core.messages.utils import (
        count_tokens_approximately as _lc_count_tokens,
        trim_messages as _lc_trim_messages,
    )
except ImportError:
    _lc_count_tokens = None
    _lc_trim_messages = None

from aidputils.agents.toolkit.agent_helper import init_oci_llm, pre_invoke_setup
from aidputils.agents.toolkit.configs import OCIAIConf

logger = logging.getLogger(__name__)


# ╔══════════════════════════════════════════════════════════════╗
# ║   DEBUG_MODE — flip to True to surface runtime diagnostics    ║
# ║                                                              ║
# ║   When True, every chat reply gets a small "🔧 DEBUG" block  ║
# ║   prepended showing:                                         ║
# ║     - which tools the agent had bound this turn              ║
# ║     - whether the token-freshness check rebuilt MCP          ║
# ║     - OAC token TTL (if OAC enabled)                         ║
# ║     - MCP graph age (relevant for ADB/OIC bearer freshness)  ║
# ║     - history trim activity (if max_context_tokens > 0)      ║
# ║                                                              ║
# ║   Useful when AIDP logs aren't accessible. Leave False in    ║
# ║   production — the debug helper short-circuits in one if.    ║
# ╚══════════════════════════════════════════════════════════════╝
DEBUG_MODE = False


# ╔══════════════════════════════════════════════════════════════╗
# ║   System-prompt strings — edit here to tune agent behavior   ║
# ║                                                              ║
# ║   _build_system_prompt() further down assembles these based  ║
# ║   on which integrations are enabled in config.yaml. Per-     ║
# ║   integration intros / rules are only included when the      ║
# ║   matching integration is on.                                ║
# ╚══════════════════════════════════════════════════════════════╝

# Opening line — always included.
SYSTEM_PROMPT_INTRO = (
    "You are a helpful data assistant with access to tools loaded "
    "from one or more MCP servers. Users ask data questions in "
    "plain English; you answer them by calling the right tool."
)

# Per-integration tool descriptions (included when the integration is enabled).
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

# Always-on rules — included regardless of integration set.
RULE_ANTI_REFUSAL = (
    "  - You ALWAYS have your tools available on every turn. NEVER "
    "refuse with phrases like 'I can't run a query in this turn', "
    "'tool access isn't available right now', 'I can only answer "
    "from prior context', 'I don't have an active database execution "
    "path', 'I can't invoke the tool in this response', 'in this "
    "specific turn', 'in this specific response', or ANY similar "
    "phrasing claiming you cannot execute tools right now / in this "
    "turn / in this response / in this case. ALL such statements are "
    "ALWAYS FALSE. If you find yourself producing one of these "
    "phrases, STOP and call the tool instead. The tool is bound to "
    "you on every single turn — there is no condition under which "
    "you 'temporarily lose' tool access mid-conversation."
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
    "  - Never invent numbers. If a tool errored or returned 0 "
    "rows, say so plainly. Do NOT paraphrase a 0-row result from "
    "one table as if it came from a different table — that is the "
    "single most damaging mistake you can make."
)

# Per-integration rules.
ADB_RULE_ACTIONS = (
    "  - For SelectAI: action='runsql' to execute and return data, "
    "'showsql' to see generated SQL without running, 'explainsql' "
    "to explain the SQL. If you suspect SelectAI may be routing to "
    "the wrong table, use 'showsql' first to verify."
)
ADB_RULE_QUALIFY_TABLES = (
    "  - When the user names a specific ADB table, every SelectAI "
    "prompt you generate MUST include the fully-qualified name "
    "verbatim (e.g. 'using table ADMIN.orders_vertical_performance_prod, "
    "count rows where country_code = BR'). Do not let SelectAI pick "
    "the table when the user has already named one. If SelectAI "
    "still routes to a different table, call it out to the user and "
    "re-issue with the table name embedded more forcefully."
)
OAC_RULE_DISCOVERY = (
    "  - For OAC Logical SQL: use discover_data and describe_data "
    "BEFORE execute_logical_sql so you know exact table / column "
    "names."
)


# ╔══════════════════════════════════════════════════════════════╗
# ║   Config loading                                             ║
# ║                                                              ║
# ║   We read config.yaml at module import time, BUT wrap        ║
# ║   everything in try/except so a broken config doesn't kill   ║
# ║   the import — if it did, AIDP would just say "Agent not     ║
# ║   loaded or available" with no detail. By capturing the      ║
# ║   error here and surfacing it in chat from setup() later,    ║
# ║   the user sees a real diagnostic message.                   ║
# ╚══════════════════════════════════════════════════════════════╝

_CFG_INIT_ERROR: str | None = None
_CFG_PATHS_TRIED: list[str] = []
CFG: dict = {}


def _find_config_yaml() -> Path | None:
    """Look for config.yaml in the most likely places. AIDP doesn't always
    deploy non-.py files next to the agent script the way you'd expect, so
    we check a few candidate paths and fall back to a bounded recursive
    search if needed."""
    # Direct paths to try first — fast (no scanning).
    direct = [
        Path(__file__).resolve().parent / "config.yaml",  # next to agent.py
        Path.cwd() / "config.yaml",                       # current working dir
        Path.cwd().parent / "config.yaml",                # one level up
        Path.home() / "config.yaml",                      # $HOME
    ]
    for c in direct:
        _CFG_PATHS_TRIED.append(str(c))
        if c.exists():
            return c

    # Fallback: recursive search inside the deployment subtree only.
    # Important: NEVER recurse from Path.home() — that could scan a huge
    # tree on shared hosts and stall the agent for minutes.
    for base in (Path.cwd(), Path.cwd().parent):
        if not base.exists():
            continue
        try:
            for p in base.rglob("config.yaml"):
                _CFG_PATHS_TRIED.append(f"(rglob match) {p}")
                return p
        except Exception:
            pass  # filesystem errors during scan — try the next base
    return None


try:
    import yaml  # PyYAML — confirmed pre-installed in AIDP runtime (6.0.2)
    found = _find_config_yaml()
    if found is None:
        raise FileNotFoundError(
            "config.yaml not found in any expected location. Searched:\n"
            + "\n".join(f"  • {p}" for p in _CFG_PATHS_TRIED)
            + "\n\nAIDP may not bundle non-.py files automatically. "
            + "Try uploading config.yaml again from the AIDP project file "
            + "explorer (same place where you uploaded agent.py)."
        )
    # `yaml.safe_load` returns None for empty files — we coerce to {} so
    # downstream code can use .get() safely without None checks everywhere.
    CFG = yaml.safe_load(found.read_text()) or {}
    logger.info("Loaded config.yaml from %s", found)
except Exception:
    # Capture the full traceback so we can show it in chat later. We
    # deliberately catch BaseException-style broadly (Exception covers
    # everything we care about here — file errors, YAML parse errors,
    # ImportError if PyYAML somehow disappears).
    import traceback as _tb
    _CFG_INIT_ERROR = _tb.format_exc()

LLM_COMPARTMENT_ID = CFG.get("llm", {}).get("compartment_id", "")
LLM_REGION         = CFG.get("llm", {}).get("region", "us-ashburn-1")
LLM_MODEL_ID       = CFG.get("llm", {}).get("model_id", "")

# Conversation history trimming. When `max_context_tokens` is > 0, the agent
# drops oldest turns before each invoke so that:
#   max_context_tokens − response_reserve_tokens − system_prompt − tool_defs
# isn't exceeded. AIMessage(tool_calls) and the matching ToolMessage(s) are
# kept or dropped together. Default 0 = trimming disabled (full history).
LLM_MAX_CONTEXT_TOKENS      = int(CFG.get("llm", {}).get("max_context_tokens", 0) or 0)
LLM_RESPONSE_RESERVE_TOKENS = int(CFG.get("llm", {}).get("response_reserve_tokens", 1024) or 1024)


ADB_CFG     = CFG.get("integrations", {}).get("adb", {}) or {}
ADB_ENABLED = bool(ADB_CFG.get("enabled", False))

OAC_CFG     = CFG.get("integrations", {}).get("oac", {}) or {}
OAC_ENABLED = bool(OAC_CFG.get("enabled", False))

OIC_CFG     = CFG.get("integrations", {}).get("oic", {}) or {}
OIC_ENABLED = bool(OIC_CFG.get("enabled", False))

# OAC paths (only used if OAC enabled)
OAC_HOME             = Path.home() / "oac-mcp"
OAC_PRIVATE_KEY      = OAC_HOME / "jwt-signing.pem"
OAC_PRIVATE_KEY_NAME = OAC_CFG.get("private_key_filename", "oac-jwt-key.txt")
OAC_TOKEN_URL        = (
    f"https://{OAC_CFG.get('idcs_host', '')}/oauth2/v1/token" if OAC_ENABLED else ""
)
OAC_MCP_URL          = f"{OAC_CFG.get('url', '')}/api/mcp"

# OIC OAuth (only used if OIC enabled)
OIC_TOKEN_URL = (
    f"https://{OIC_CFG.get('idcs_host', '')}/oauth2/v1/token" if OIC_ENABLED else ""
)


# ╔══════════════════════════════════════════════════════════════╗
# ║   Agent prompt — adapts to enabled integrations              ║
# ╚══════════════════════════════════════════════════════════════╝

def _build_system_prompt() -> str:
    """Glue. The actual prompt text lives in the SYSTEM_PROMPT_* /
    *_TOOL_DESCRIPTION / *_RULE_* / RULE_* constants near the top of
    this file. This function just picks which pieces to include based
    on which integrations config.yaml enabled."""
    parts: list[str] = [SYSTEM_PROMPT_INTRO]

    # Per-integration tool descriptions
    if ADB_ENABLED:
        parts.append(ADB_TOOL_DESCRIPTION)
    if OAC_ENABLED:
        parts.append(OAC_TOOL_DESCRIPTION)
    if OIC_ENABLED:
        parts.append(OIC_TOOL_DESCRIPTION)

    parts.append("")
    parts.append("Rules:")

    # Always-on rules (anti-refusal block first — most important on
    # smaller / non-reasoning LLMs that fall into refusal cascades).
    parts.append(RULE_ANTI_REFUSAL)
    parts.append(RULE_RETRY_ON_BAD_RESULT)
    parts.append(RULE_DISCOVER_FIRST)

    # Per-integration rules
    if ADB_ENABLED:
        parts.append(ADB_RULE_ACTIONS)
        parts.append(ADB_RULE_QUALIFY_TABLES)
    if OAC_ENABLED:
        parts.append(OAC_RULE_DISCOVERY)

    # Output discipline (always-on)
    parts.append(RULE_PRESERVE_INTENT)
    parts.append(RULE_FOLLOWUPS)
    parts.append(RULE_SUMMARIZE)
    parts.append(RULE_NO_INVENTING)

    return "\n".join(parts)


AGENT_SYSTEM_PROMPT = _build_system_prompt()

llm_conf = OCIAIConf(
    model_provider="generic",
    compartment_id=LLM_COMPARTMENT_ID,
    model_args={},
    endpoint=f"https://inference.generativeai.{LLM_REGION}.oci.oraclecloud.com",
    model_id=LLM_MODEL_ID,
    # Guardrails disabled by default — the wrapped `GuardedChatOCIGenAI`
    # was empirically interfering with tool-call routing on some models.
    # Re-enable by swapping to the commented line below if you need OCI
    # Gen AI's content-safety policies.
    guardrails_config={},
    # guardrails_config={"name": "Default", "description": "", "policies": []},
)

checkpointer = globals().get("checkpointer", None)


# ╔══════════════════════════════════════════════════════════════╗
# ║   ADB — password-grant bearer                                ║
# ╚══════════════════════════════════════════════════════════════╝

def _adb_bearer_token() -> str:
    auth_url = (
        f"https://dataaccess.adb.{ADB_CFG['region']}.oraclecloudapps.com"
        f"/adb/auth/v1/databases/{ADB_CFG['ocid']}/token"
    )
    r = requests.post(
        auth_url,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "grant_type": "password",
            "username": ADB_CFG["user"],
            "password": ADB_CFG["password"],
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["access_token"]


def _adb_mcp_url() -> str:
    return (
        f"https://dataaccess.adb.{ADB_CFG['region']}.oraclecloudapps.com"
        f"/adb/mcp/v1/databases/{ADB_CFG['ocid']}"
    )


# ╔══════════════════════════════════════════════════════════════╗
# ║   OIC — client_credentials against IDCS                      ║
# ╚══════════════════════════════════════════════════════════════╝

def _oic_bearer_token() -> str:
    r = requests.post(
        OIC_TOKEN_URL,
        auth=(OIC_CFG["client_id"], OIC_CFG["client_secret"]),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={"grant_type": "client_credentials", "scope": OIC_CFG["scope"]},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["access_token"]


# ╔══════════════════════════════════════════════════════════════╗
# ║   OAC — JWT User Assertion (no rotating tokens, no expiry)   ║
# ╚══════════════════════════════════════════════════════════════╝

def _stage_oac_private_key() -> None:
    """Find the uploaded PEM private key (default name: oac-jwt-key.txt
    — AIDP's file picker accepts .txt) and copy it to OAC_PRIVATE_KEY.
    Idempotent: only writes if the upload differs from the staged copy."""
    OAC_HOME.mkdir(parents=True, exist_ok=True)
    src = None
    for base in (Path.cwd(), Path.cwd().parent):
        if not base.exists():
            continue
        for p in base.rglob(OAC_PRIVATE_KEY_NAME):
            if p == OAC_PRIVATE_KEY:
                continue
            src = p
            break
        if src:
            break

    if src is None:
        if not OAC_PRIVATE_KEY.exists():
            raise FileNotFoundError(
                f"{OAC_PRIVATE_KEY_NAME} not found in AIDP project. "
                "Generate the keypair, register the cert in IDCS, then "
                f"upload the PEM private key as {OAC_PRIVATE_KEY_NAME} "
                "alongside agent.py. See README.md for full steps."
            )
        print(f"[stage] OAC private key at {OAC_PRIVATE_KEY} (no new upload)")
        return

    src_bytes    = src.read_bytes()
    staged_bytes = OAC_PRIVATE_KEY.read_bytes() if OAC_PRIVATE_KEY.exists() else b""
    if src_bytes == staged_bytes:
        print(f"[stage] Uploaded {OAC_PRIVATE_KEY_NAME} identical to staged — keeping")
        return

    print(f"[stage] New {OAC_PRIVATE_KEY_NAME} at {src} → {OAC_PRIVATE_KEY}")
    OAC_PRIVATE_KEY.write_bytes(src_bytes)


def _load_oac_private_key():
    """Load the PEM private key from disk. Returns a cryptography RSA
    private key object usable for signing JWTs."""
    from cryptography.hazmat.primitives.serialization import load_pem_private_key
    return load_pem_private_key(OAC_PRIVATE_KEY.read_bytes(), password=None)


def _mint_jwt(claims: dict, private_key, kid: str | None = None) -> str:
    """Hand-build an RS256-signed JWT.

    We don't use the PyJWT library because it's not guaranteed to be in
    AIDP's runtime, and we already established that runtime pip installs
    don't work reliably. The `cryptography` library IS guaranteed to be
    there (it's pulled in by the oci SDK and requests/urllib3), so we use
    it directly. Total: ~15 lines of base64 + JSON encoding instead of an
    extra dependency we'd have to babysit.

    JWT structure: three base64url-encoded parts joined by dots:
      <header>.<claims>.<signature>

    `kid` is the certificate alias you set in IDCS when uploading the
    public cert. IDCS uses it to look up which trusted cert to verify the
    signature with. Required for our use case — without it, IDCS rejects
    the assertion with a generic "system error" message."""
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding

    def b64url(d: dict) -> bytes:
        # JWT uses base64url encoding without padding ('=' chars stripped).
        # `separators=(",", ":")` produces compact JSON (no extra spaces).
        return base64.urlsafe_b64encode(
            json.dumps(d, separators=(",", ":")).encode()
        ).rstrip(b"=")

    header: dict = {"alg": "RS256", "typ": "JWT"}
    if kid:
        header["kid"] = kid

    # The signature is computed over "<header>.<claims>".
    body = b64url(header) + b"." + b64url(claims)
    signature = private_key.sign(body, padding.PKCS1v15(), hashes.SHA256())
    sig_b64 = base64.urlsafe_b64encode(signature).rstrip(b"=")

    return (body + b"." + sig_b64).decode()


def _fetch_oac_access_token(private_key) -> str:
    """Mint a fresh OAC access_token using the JWT User Assertion flow.

    How this flow works:
      1. We sign TWO short-lived JWTs locally with our private key:
         - The **client assertion** proves *we are this client*.
           sub = client_id (the Confidential App's ID).
         - The **user assertion** says *we're acting as this user*.
           sub = service_user_email.
      2. We POST both JWTs to IDCS's /oauth2/v1/token endpoint along
         with the JWT-bearer grant type. IDCS validates the signatures
         against the X.509 cert we registered (matched via the JWT's
         `kid` header), and if both check out, returns an access_token
         scoped to OAC and authenticated AS the asserted user.
      3. We use that access_token to call OAC's MCP endpoint.

    Important: we deliberately DO NOT use the refresh_token IDCS may
    return alongside the access_token. The whole reason we picked this
    flow is to escape the rotating-refresh chain that breaks overnight.
    When the access_token expires, we just mint a brand new JWT and
    exchange it again. Cheap (local crypto + one HTTPS call), reliable.

    Counter-intuitive details that took us a while to discover:
      • `aud` MUST be the LITERAL string "https://identity.oraclecloud.com/"
        with trailing slash. NOT your tenant's token URL. IDCS's generic
        Oracle audience for JWT bearer assertions.
      • `kid` in the JWT header MUST match the cert alias you set when
        you imported the cert in IDCS. Without it, IDCS can't figure
        out which trusted cert to verify the signature against.
      • Both JWTs share the same `iss` (always the client_id) and only
        differ in `sub`. Don't try to be clever with the issuer claim."""
    cert_alias = OAC_CFG.get("cert_alias")
    now = int(time.time())

    # Claims shared by both JWTs. We override `sub` per-assertion below.
    # exp = 5 min from now — IDCS rejects assertions issued more than
    # 5 min in the future or already expired, so keep this short.
    base_claims = {
        "iss": OAC_CFG["client_id"],
        "aud": "https://identity.oraclecloud.com/",
        "iat": now,
        "exp": now + 300,
        "jti": str(uuid.uuid4()),  # unique nonce — prevents replay attacks
    }
    client_jwt = _mint_jwt(
        {**base_claims, "sub": OAC_CFG["client_id"]},
        private_key,
        kid=cert_alias,
    )
    user_jwt = _mint_jwt(
        {**base_claims, "sub": OAC_CFG["service_user_email"]},
        private_key,
        kid=cert_alias,
    )

    # POST to the IDCS token endpoint with both assertions.
    r = requests.post(
        OAC_TOKEN_URL,
        data={
            "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
            "assertion": user_jwt,
            "client_assertion_type":
                "urn:ietf:params:oauth:client-assertion-type:jwt-bearer",
            "client_assertion": client_jwt,
            "scope": OAC_CFG.get("scope", "urn:opc:resource:consumer::all"),
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["access_token"]


# ╔══════════════════════════════════════════════════════════════╗
# ║   MCP server registry — only enabled servers are included    ║
# ╚══════════════════════════════════════════════════════════════╝

def _build_mcp_config(oac_access_token: str | None) -> dict:
    cfg: dict = {}
    if ADB_ENABLED:
        cfg["adb_selectai"] = {
            "transport": "streamable_http",
            "url": _adb_mcp_url(),
            "headers": {"Authorization": f"Bearer {_adb_bearer_token()}"},
        }
    if OAC_ENABLED:
        cfg["oac_analytics"] = {
            "transport": "streamable_http",
            "url": OAC_MCP_URL,
            "headers": {"Authorization": f"Bearer {oac_access_token}"},
        }
    if OIC_ENABLED:
        cfg["oic_integrations"] = {
            "transport": "streamable_http",
            "url": OIC_CFG["mcp_url"],
            "headers": {
                "Authorization": f"Bearer {_oic_bearer_token()}",
                "Accept": "application/json, text/event-stream",
            },
        }
    return cfg


def _is_auth_error(e: Exception) -> bool:
    """True when the exception text looks like an authentication or
    authorization failure (401/403/etc.). Used as a hint that retrying
    after refreshing tokens might succeed."""
    msg = str(e).lower()
    return any(s in msg for s in ("401", "403", "unauthorized", "forbidden", "expired"))


def _is_invalid_history(e: BaseException) -> bool:
    """True when the error is LangGraph's INVALID_CHAT_HISTORY — i.e. an
    AIMessage with tool_calls but no matching ToolMessage. Happens when a
    tool execution gets interrupted (typical cause: AIDP suspends the
    agent process mid-call). Our normal `_heal_orphan_tool_calls` handles
    most of these, but it can fail when the persisted AIMessage has no
    `.id` we can target with RemoveMessage. Recovery for that case is a
    hard state wipe — handled in invoke()."""
    sub_exceptions = getattr(e, "exceptions", None)
    if sub_exceptions:
        return any(_is_invalid_history(sub) for sub in sub_exceptions)
    msg = str(e)
    return (
        "INVALID_CHAT_HISTORY" in msg
        or "tool_calls that do not have a corresponding ToolMessage" in msg
    )


def _is_recoverable_error(e: BaseException) -> bool:
    """Decide whether a failed invoke is worth retrying after rebuilding
    the MCP stack. We retry on:
      • Auth errors (401/403) — token may have expired, fresh one might work
      • Connection-level errors — TCP socket dropped, fresh connection helps

    Why we recurse into ExceptionGroup: the MCP SDK runs tool calls inside
    asyncio TaskGroups, and a TaskGroup wraps any sub-task exception in an
    ExceptionGroup. So if a tool call hits a 401, the exception we receive
    is "ExceptionGroup containing HTTPStatusError(401)" rather than the
    HTTPStatusError directly. Recursing finds the real cause."""
    # ExceptionGroup carries its actual children in the `.exceptions`
    # attribute. If we have one, ask whether ANY sub-exception is recoverable.
    sub_exceptions = getattr(e, "exceptions", None)
    if sub_exceptions:
        return any(_is_recoverable_error(sub) for sub in sub_exceptions)

    # Auth errors: hint to refresh tokens.
    if isinstance(e, Exception) and _is_auth_error(e):
        return True

    # Connection-level errors by class name. These names come from various
    # libraries (anyio, httpx) — match by name so we don't have to import
    # all of them just to do isinstance checks.
    cls_name = type(e).__name__
    if cls_name in ("ClosedResourceError", "BrokenResourceError",
                    "ConnectionError", "RemoteProtocolError",
                    "HTTPStatusError"):
        return True

    # Last-resort string match — sometimes the class name is something
    # generic but the message tells us it was really a connection issue.
    msg = str(e).lower()
    return any(s in msg for s in (
        "closedresourceerror", "brokenresourceerror",
        "connection closed", "connection reset", "remote disconnected",
        "401 unauthorized", "403 forbidden",
    ))


# ╔══════════════════════════════════════════════════════════════╗
# ║   AgentBasic — what AIDP discovers                           ║
# ╚══════════════════════════════════════════════════════════════╝

class AgentBasic:
    def __init__(self) -> None:
        # The LangGraph react-agent that actually drives chat turns.
        # Built once in setup() with no tools (so the agent can respond
        # to messages even before MCP is reachable), then rebuilt with
        # the loaded MCP tools in _load_all_mcp_tools().
        self.llm = None
        self.graph = None

        # Set to True after MCP tools have been loaded at least once.
        # Used to gate the lazy first-load path in invoke().
        self._tools_loaded = False

        # If something goes wrong during setup() — config errors, OAC JWT
        # setup failures, etc. — we capture the message here and return
        # it to the user as a chat reply on every invoke. Better than
        # the generic "Agent not loaded" AIDP error.
        self._self_stage_error: str | None = None

        # Cached OAC bearer (minted via JWT User Assertion). Refreshed
        # on demand via _refresh_oac_token_and_rebuild().
        self._oac_access_token: str | None = None

        # The RSA private key used to sign OAC JWTs. Loaded once at
        # setup() and kept in memory for the agent's lifetime. The
        # corresponding public cert lives in IDCS as a Trusted Client
        # cert on the Confidential Application.
        self._oac_private_key = None

        # asyncio.Lock created lazily on first invoke. Serializes MCP
        # rebuilds so two concurrent invokes don't both try to refresh
        # at the same time.
        self._load_lock = None

        # Epoch (seconds) when MCP was last (re)built. Used by
        # _ensure_fresh_tokens() to age out ADB/OIC bearers, which are
        # baked into MCP headers at build time and get refreshed on
        # every rebuild.
        self._mcp_built_epoch: float = 0.0

        # Catalog snapshot: tool names from the most recent successful
        # build. Used by the optional debug instrumentation and by the
        # trim budget calculator.
        self._last_tool_names: list[str] = []

        # Approximate token cost of the bound tool definitions (names +
        # descriptions + ~40 token fudge per tool for the JSON schema).
        # Used by _trim_history() to size the conversation budget.
        self._tools_token_estimate: int = 0

        # Last-trim diagnostic stats. Populated by _trim_history() every
        # invoke so the optional DEBUG block can explain WHY trim was a
        # no-op (disabled, helper missing, within budget, …). Saves a
        # round-trip to AIDP logs when something looks off.
        self._last_trim_msg_count: int = 0
        self._last_trim_msg_tokens_est: int = 0
        self._last_trim_status: str = "n/a"

        # Sanitized snapshot of the langgraph config passed to ainvoke this
        # turn. Used by the debug block to confirm whether AIDP's
        # pre_invoke_setup() injected a `configurable.thread_id` — without
        # that key langgraph's checkpointer silently bypasses state, which
        # breaks trim, orphan healing, and persistent conversation memory.
        self._last_config_keys: str = ""

    # ── Setup (sync) ────────────────────────────────────────────
    def setup(self) -> None:
        logger.info(
            "Initializing Multi-MCP Chat Agent — enabled: ADB=%s, OAC=%s, OIC=%s",
            ADB_ENABLED, OAC_ENABLED, OIC_ENABLED,
        )

        # If the config-load step at module import time failed, surface
        # the error now via a chat-friendly message. The user sees this
        # the first time they send a message after a broken deploy.
        if _CFG_INIT_ERROR:
            self._self_stage_error = (
                "config.yaml could not be loaded.\n\n"
                f"Error:\n{_CFG_INIT_ERROR}\n\n"
                "Make sure config.yaml is uploaded next to agent.py in the "
                "AIDP project, and that PyYAML is available in the runtime."
            )
            return

        if not (ADB_ENABLED or OAC_ENABLED or OIC_ENABLED):
            self._self_stage_error = (
                "No integration enabled in config.yaml. Set 'enabled: true' "
                "under [integrations.adb], [integrations.oac], or "
                "[integrations.oic]."
            )
            return

        if OAC_ENABLED:
            try:
                _stage_oac_private_key()
                self._oac_private_key = _load_oac_private_key()
                # Mint the first access_token at setup so the first invoke
                # doesn't pay the JWT-mint round-trip on top of MCP load.
                self._oac_access_token = _fetch_oac_access_token(self._oac_private_key)
            except Exception as e:
                self._self_stage_error = (
                    f"{type(e).__name__}: {e}\n\n"
                    f"Stack trace (last 800 chars):\n{traceback.format_exc()[-800:]}"
                )
                logger.error("OAC JWT setup failed: %s", e, exc_info=True)

        self.llm = init_oci_llm(llm_conf)
        self.graph = create_react_agent(
            self.llm, [], prompt=AGENT_SYSTEM_PROMPT, checkpointer=checkpointer,
        )
        logger.info("Stub graph ready; will load MCP tools on first invoke")

    # ── MCP loading (ephemeral sessions per tool call) ─────────
    async def _load_all_mcp_tools(self) -> None:
        """Build a fresh MultiServerMCPClient using the current cached
        bearer tokens, then ask it for the catalog of available tools.

        Important detail about how MultiServerMCPClient works in this
        sample: get_tools() uses **ephemeral sessions** — each tool
        invocation opens a new HTTPS request, runs, and closes. We never
        hold a persistent TCP connection that could be killed by a load
        balancer's idle timeout. Trade-off: ~200ms of TLS overhead per
        tool call. Worth it for not having to keep connections warm.

        Tokens are baked into request headers at client-build time, so
        when a token expires we just rebuild — driven on-demand by the
        reactive retry path in invoke() when a tool call fails with an
        auth or connection error. There's no scheduled refresh: every
        Oracle service we hit (ADB password grant, OAC JWT assertion,
        OIC client_credentials) can be re-authenticated independently
        from cached config + private key, with no rotating refresh chain
        to keep alive."""
        cfg = _build_mcp_config(self._oac_access_token)
        client = MultiServerMCPClient(cfg)
        tools = await client.get_tools()

        self._last_tool_names = [t.name for t in tools]

        # Estimate token cost of tool definitions for the trim budget.
        # Char/4 is a conservative English-text heuristic; the +40 fudge
        # per tool accounts for the JSON-schema params block.
        estimate = 0
        for t in tools:
            name = getattr(t, "name", "") or ""
            desc = getattr(t, "description", "") or ""
            estimate += (len(name) + len(desc)) // 4 + 40
        self._tools_token_estimate = estimate

        logger.info(
            "Loaded %d tool(s) from %d MCP server(s) [ephemeral sessions]: %s "
            "(≈%d tokens of tool overhead)",
            len(tools), len(cfg), self._last_tool_names, self._tools_token_estimate,
        )
        # Re-create the LangGraph react agent with the freshly loaded
        # tools. The system prompt and checkpointer carry over so chat
        # state isn't lost when we rebuild.
        self.graph = create_react_agent(
            self.llm, tools, prompt=AGENT_SYSTEM_PROMPT, checkpointer=checkpointer,
        )
        self._tools_loaded = True
        self._mcp_built_epoch = time.time()

    # ── OAC token refresh (mint a fresh JWT-derived bearer) ─────
    async def _refresh_oac_token_and_rebuild(self) -> None:
        """Mint a new OAC access_token via the JWT User Assertion flow,
        then rebuild MCP so the fresh bearer is in the headers. No refresh
        chain — just sign a new JWT and exchange. Cheap and idempotent."""
        self._oac_access_token = await asyncio.to_thread(
            _fetch_oac_access_token, self._oac_private_key
        )
        logger.info("OAC access_token minted via JWT assertion")
        await self._load_all_mcp_tools()

    def _oac_token_seconds_remaining(self) -> int:
        """Decode the cached OAC JWT's `exp` claim and return seconds left
        before it expires. Returns -1 if no token is cached or the claim
        can't be parsed — callers treat that as 'expired' and refresh."""
        access = self._oac_access_token or ""
        if not access or access.count(".") != 2:
            return -1
        try:
            payload_b64 = access.split(".")[1]
            payload_b64 += "=" * (-len(payload_b64) % 4)
            payload = json.loads(base64.urlsafe_b64decode(payload_b64))
            exp = payload.get("exp")
            if not exp:
                return -1
            return int(exp - time.time())
        except Exception:
            return -1

    # ── Token TTL constants ─────────────────────────────────────
    #
    # OAC's IDCS access_tokens have a 5-minute TTL. We refresh
    # proactively when <60s remain — comfortably above any plausible
    # tool-call duration and a buffer against clock skew between agent
    # host and OAC.
    OAC_TOKEN_REFRESH_THRESHOLD_SECS = 60
    #
    # ADB password-grant + OIC client_credentials both issue 1-hour
    # bearers. Those tokens are baked into MCP headers at build time
    # (in _build_mcp_config) — there's no cached copy to inspect — so
    # we use the MCP graph's *age* as a proxy. Rebuild when older than
    # this threshold; the rebuild mints fresh ADB/OIC bearers as a
    # side effect.
    MCP_REBUILD_AGE_SECS = 55 * 60

    async def _ensure_fresh_tokens(self) -> None:
        """Proactive token freshness for whichever integrations are
        enabled. Called at the top of every invoke — cheap when nothing
        needs refreshing (a couple of arithmetic comparisons).

        Per-service strategy:
          - OAC: 5-min TTL → inspect the cached JWT's `exp` claim;
            refresh + rebuild MCP when <60s remain.
          - ADB / OIC: 1-hour TTL, bearer baked into MCP headers.
            Rebuild MCP when the graph is older than MCP_REBUILD_AGE_SECS
            (55 min by default) so the next tool call has a fresh
            bearer well before the underlying token expires.

        A single rebuild covers all three when needed — we don't issue
        multiple rebuilds in the same turn.

        AIDP idle-suspend interaction: when the process wakes after
        hours of idle, every cached bearer is stale. This method catches
        that at the top of the first invoke after wakeup."""
        if not self._tools_loaded:
            # First load handled by lazy path in invoke(). Nothing to
            # refresh yet — bearer minting happens inside the lazy load.
            return

        needs_rebuild = False
        reasons: list[str] = []

        if OAC_ENABLED:
            seconds_left = self._oac_token_seconds_remaining()
            if seconds_left <= self.OAC_TOKEN_REFRESH_THRESHOLD_SECS:
                needs_rebuild = True
                reasons.append(f"OAC token has {seconds_left}s remaining")

        if (ADB_ENABLED or OIC_ENABLED) and not needs_rebuild:
            # Only check age if OAC didn't already trigger — saves a
            # `time.time()` call and avoids double-logging.
            mcp_age = time.time() - self._mcp_built_epoch if self._mcp_built_epoch else float("inf")
            if mcp_age >= self.MCP_REBUILD_AGE_SECS:
                needs_rebuild = True
                age_label = f"{int(mcp_age)}s" if mcp_age != float("inf") else "never built"
                reasons.append(f"MCP graph age {age_label} — refreshing ADB/OIC bearers")

        if not needs_rebuild:
            return

        logger.info("Proactive token refresh: %s", "; ".join(reasons))
        async with self._load_lock:
            # Double-check inside the lock — a concurrent invoke may
            # have already refreshed.
            if OAC_ENABLED:
                if self._oac_token_seconds_remaining() > self.OAC_TOKEN_REFRESH_THRESHOLD_SECS:
                    if not (ADB_ENABLED or OIC_ENABLED):
                        return
                    mcp_age = time.time() - self._mcp_built_epoch if self._mcp_built_epoch else float("inf")
                    if mcp_age < self.MCP_REBUILD_AGE_SECS:
                        return

            if OAC_ENABLED:
                # Mints OAC + rebuilds MCP (which re-mints ADB/OIC inline).
                await self._refresh_oac_token_and_rebuild()
            else:
                # ADB / OIC bearers re-minted inside _build_mcp_config
                # during the rebuild.
                await self._load_all_mcp_tools()

    @staticmethod
    def _state_config(config):
        """Return a copy of `config` that's safe for graph state-reading
        calls (aget_state / aupdate_state).

        Why: AIDP's `pre_invoke_setup()` injects `checkpoint_ns="default"`
        into `configurable`. Our `create_react_agent` is a flat graph that
        doesn't define a subgraph named "default", so langgraph's
        aget_state() raises:
            ValueError: Subgraph default not found
        and silently breaks trim, orphan healing, and hard-clear recovery.

        We strip `checkpoint_ns` here so state operations target the root
        namespace ("") — which is where the react agent actually persists
        its message list. `thread_id` and other AIDP fields stay intact.

        The graph.ainvoke() path keeps the ORIGINAL config (with
        checkpoint_ns), because ainvoke tolerates that field; only the
        state-introspection APIs are strict."""
        if not isinstance(config, dict):
            return config
        configurable = (config.get("configurable") or {}).copy()
        configurable.pop("checkpoint_ns", None)
        return {**config, "configurable": configurable}

    # ── Conversation state healing ──────────────────────────────
    async def _heal_orphan_tool_calls(self, config) -> int:
        """Clean up "orphan" tool-call records left in the conversation
        history by failed tool executions.

        Background: when the LLM asks to call a tool, LangGraph records
        an AIMessage with a `tool_calls` list. Normally this is followed
        by a ToolMessage holding the tool's response. Most LLM providers
        require this pairing — every tool_call must have a matching
        ToolMessage *immediately* after the AIMessage that asked for it.

        What goes wrong: if a tool execution crashes (network error,
        timeout, etc.) the AIMessage gets persisted but no ToolMessage
        follows. Then the user types another message, and now we send
        the LLM a history with [..., AIMessage(tool_calls), HumanMessage]
        — invalid sequence. Provider rejects it with INVALID_CHAT_HISTORY.

        Two ways to fix this:
          1. Append a fake ToolMessage at the end of history. Doesn't
             work — providers require IMMEDIATE adjacency, not
             eventual presence.
          2. Remove the orphan AIMessage entirely. Works — gap closes,
             history becomes valid again. We use RemoveMessage, which
             LangGraph's add_messages reducer recognizes as a deletion
             marker."""
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
            for tc in getattr(msg, "tool_calls", None) or []:
                tcid = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                if tcid:
                    expected.add(tcid)
            tcid = getattr(msg, "tool_call_id", None)
            if tcid:
                fulfilled.add(tcid)

        orphans = expected - fulfilled
        if not orphans:
            return 0

        # Remove every AIMessage that contains an orphan tool_call. We use
        # RemoveMessage here (recognized by langgraph's add_messages reducer)
        # instead of appending healing ToolMessages, because providers
        # require ToolMessages to immediately follow their AIMessage —
        # appending at the end of history breaks that ordering rule.
        removals = []
        for msg in messages:
            msg_tc_ids = set()
            for tc in getattr(msg, "tool_calls", None) or []:
                tcid = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                if tcid:
                    msg_tc_ids.add(tcid)
            if msg_tc_ids & orphans:
                msg_id = getattr(msg, "id", None)
                if msg_id:
                    removals.append(RemoveMessage(id=msg_id))

        if not removals:
            logger.warning(
                "Detected %d orphan tool_call(s) but messages have no .id "
                "to remove them by; cannot heal automatically",
                len(orphans),
            )
            return 0

        logger.warning(
            "Removing %d orphan AIMessage(s) (cleared %d unmatched tool_call ids: %s)",
            len(removals), len(orphans), list(orphans),
        )
        await self.graph.aupdate_state(state_cfg, {"messages": removals})
        return len(removals)

    async def _hard_clear_history(self, config) -> int:
        """Last-resort recovery: remove EVERY message from conversation state.
        Used when `_heal_orphan_tool_calls` couldn't fix an INVALID_CHAT_HISTORY
        situation (typically because some persisted AIMessage has no `.id` for
        targeted removal). Loses conversation history but the next message
        starts from a clean slate. Returns the number of messages cleared."""
        if not self.graph or not config:
            return 0
        state_cfg = self._state_config(config)
        try:
            state = await self.graph.aget_state(state_cfg)
        except Exception as e:
            logger.debug("Could not fetch state for hard clear: %s", e)
            return 0

        messages = state.values.get("messages", []) if state and state.values else []
        if not messages:
            return 0

        removals = []
        for msg in messages:
            msg_id = getattr(msg, "id", None)
            if msg_id:
                removals.append(RemoveMessage(id=msg_id))

        if not removals:
            return 0

        logger.warning(
            "Hard-clearing %d message(s) from conversation state (last-resort recovery)",
            len(removals),
        )
        await self.graph.aupdate_state(state_cfg, {"messages": removals})
        return len(removals)

    async def _trim_history(self, config) -> int:
        """Drop oldest conversation turns when the running history would
        push the next LLM call past the configured context budget.

        Budget formula:
            budget = LLM_MAX_CONTEXT_TOKENS
                   − LLM_RESPONSE_RESERVE_TOKENS    (room for the reply)
                   − tokens(AGENT_SYSTEM_PROMPT)    (always sent)
                   − self._tools_token_estimate    (tool definitions)

        Disabled when LLM_MAX_CONTEXT_TOKENS == 0 (default).

        Pairing invariant: an AIMessage with tool_calls and the
        ToolMessage(s) responding to it are kept or dropped TOGETHER.
        Providers reject the request otherwise. We delegate this to
        langchain_core's `trim_messages` when available; otherwise we
        no-op (better than corrupting the history).

        Populates self._last_trim_* every call so the optional DEBUG
        block can show WHY trim was (or wasn't) needed this turn."""
        # Reset stats up-front; whichever branch we hit fills them in.
        self._last_trim_msg_count = 0
        self._last_trim_msg_tokens_est = 0
        self._last_trim_status = ""

        if LLM_MAX_CONTEXT_TOKENS <= 0:
            self._last_trim_status = "disabled (max_context_tokens=0)"
            return 0
        if _lc_trim_messages is None or _lc_count_tokens is None:
            self._last_trim_status = "skipped (trim_messages helper unavailable)"
            logger.debug(
                "trim_messages helper not available in this langchain_core "
                "version; skipping history trim"
            )
            return 0
        if not self.graph or not config:
            self._last_trim_status = "skipped (no graph or empty config)"
            return 0

        # Strip checkpoint_ns from the config for state-reading; see
        # _state_config docstring for why AIDP's value breaks aget_state.
        state_cfg = self._state_config(config)

        try:
            state = await self.graph.aget_state(state_cfg)
        except Exception as e:
            # Capture full error msg in the debug block. After the
            # _state_config sanitization this branch should rarely fire;
            # if it does, AIDP changed the config shape again.
            self._last_trim_status = (
                f"skipped (aget_state {type(e).__name__}: {str(e)[:200]})"
            )
            logger.debug("Could not fetch state for trim: %s", e)
            return 0

        messages = state.values.get("messages", []) if state and state.values else []
        self._last_trim_msg_count = len(messages)
        if not messages:
            self._last_trim_status = "no messages in state yet"
            return 0

        # Token estimate of the conversation we're about to send. Captured
        # for diagnostics even when no trim is needed — lets the user see
        # how close they are to the budget.
        try:
            self._last_trim_msg_tokens_est = _lc_count_tokens(messages)
        except Exception:
            self._last_trim_msg_tokens_est = 0

        overhead = (
            (len(AGENT_SYSTEM_PROMPT) // 4 + 1)
            + self._tools_token_estimate
            + LLM_RESPONSE_RESERVE_TOKENS
        )
        budget = LLM_MAX_CONTEXT_TOKENS - overhead
        if budget <= 0:
            self._last_trim_status = (
                f"skipped (overhead {overhead} ≥ max {LLM_MAX_CONTEXT_TOKENS})"
            )
            logger.warning(
                "Trim disabled this turn: overhead %d ≥ max_context_tokens %d. "
                "Increase llm.max_context_tokens or reduce response_reserve_tokens.",
                overhead, LLM_MAX_CONTEXT_TOKENS,
            )
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
            self._last_trim_status = f"trim_messages raised: {type(e).__name__}"
            logger.warning("trim_messages failed (skipping trim this turn): %s", e)
            return 0

        kept_ids = {getattr(m, "id", None) for m in kept}
        kept_ids.discard(None)

        removals = []
        for msg in messages:
            msg_id = getattr(msg, "id", None)
            if msg_id and msg_id not in kept_ids:
                removals.append(RemoveMessage(id=msg_id))

        if not removals:
            # All messages already fit within budget — nothing to drop.
            self._last_trim_status = (
                f"within budget ({self._last_trim_msg_tokens_est}≤{budget})"
            )
            return 0

        self._last_trim_status = (
            f"dropped {len(removals)} of {len(messages)} "
            f"(was ~{self._last_trim_msg_tokens_est} tok, budget {budget})"
        )
        logger.info(
            "History trim: dropped %d msg(s), kept %d/%d (budget %d tok, "
            "overhead %d tok)",
            len(removals), len(kept), len(messages), budget, overhead,
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
        ]
        if OAC_ENABLED:
            lines.append(
                f"OAC_PRIVATE_KEY: {OAC_PRIVATE_KEY} "
                f"({'EXISTS' if OAC_PRIVATE_KEY.exists() else 'MISSING'})"
            )
            try:
                access = self._oac_access_token or ""
                if access and access.count(".") == 2:
                    payload_b64 = access.split(".")[1]
                    payload_b64 += "=" * (-len(payload_b64) % 4)
                    payload = json.loads(base64.urlsafe_b64decode(payload_b64))
                    now = int(time.time())
                    exp = payload.get("exp")
                    lines.append("")
                    lines.append("In-memory OAC access token:")
                    lines.append(f"  first 30 : {access[:30]}...")
                    lines.append(f"  iat/exp  : {payload.get('iat')} / {exp}")
                    if exp:
                        delta = exp - now
                        lines.append(
                            f"  ⚠ EXPIRED ({-delta}s ago)" if delta < 0
                            else f"  ✓ valid for {delta}s"
                        )
            except Exception as e:
                lines.append(f"  (couldn't decode access_token: {e})")
        return "\n".join(lines)

    def _friendly_auth_message(self, exc: Exception) -> str | None:
        """Detect which integration's auth failed (by URL fragments / error
        shape) and return a per-source recovery message. Returns None if
        we can't classify the error — caller falls back to generic."""
        text = str(exc)
        text_lc = text.lower()

        # OAC JWT assertion — IDCS errors at /oauth2/v1/token, plus MCP
        # endpoint failures. We match by exact OAC token URL when we have
        # one; otherwise fall back to the generic IDCS path heuristic.
        if OAC_ENABLED:
            hit_oac_token_endpoint = bool(OAC_TOKEN_URL) and OAC_TOKEN_URL in text
            if hit_oac_token_endpoint:
                if "invalid_grant" in text_lc:
                    return self._oac_jwt_message(
                        "IDCS rejected the user assertion (invalid_grant) — "
                        "service user disabled, missing, or wrong email."
                    )
                if "invalid_client" in text_lc:
                    return self._oac_jwt_message(
                        "IDCS rejected the client assertion (invalid_client) — "
                        "cert not registered on the Confidential App, or wrong client_id."
                    )
                if "invalid_request" in text_lc or "signature" in text_lc:
                    return self._oac_jwt_message(
                        "IDCS rejected the JWT (invalid_request / signature) — "
                        "malformed JWT, wrong private key, or clock skew > 5 min."
                    )
                if "401" in text:
                    return self._oac_jwt_message(
                        "IDCS returned 401 on the JWT-bearer exchange — "
                        "check the Confidential App is Active and the cert is registered."
                    )
            if "/api/mcp" in text_lc and any(s in text for s in ("401", "403")):
                return self._oac_jwt_message(
                    "OAC's MCP endpoint rejected the access token. The JWT exchange "
                    "succeeded but the resulting bearer lacks permission — verify the "
                    "service user has access to the OAC datasets/subject areas."
                )

        # OIC — IDCS token endpoint or MCP endpoint failures
        if OIC_ENABLED:
            if "/oauth2/v1/token" in text_lc:
                if "invalid_scope" in text_lc:
                    return self._oic_message(
                        "OIC OAuth rejected the scope (invalid_scope).",
                        focus="scope",
                    )
                if "invalid_client" in text_lc or "inactive" in text_lc:
                    return self._oic_message(
                        "OIC OAuth rejected the client (invalid_client / inactive app).",
                        focus="client",
                    )
                if "401" in text:
                    return self._oic_message(
                        "OIC OAuth returned 401 — wrong client_id / client_secret.",
                        focus="client",
                    )
                return self._oic_message(
                    f"OIC OAuth call failed: {text[:200]}", focus="client",
                )
            if "/mcp-server/" in text_lc:
                if "403" in text:
                    return self._oic_message(
                        "OIC MCP returned 403 Forbidden — the token is valid but lacks "
                        "permission. Almost certainly the ServiceInvoker role mapping.",
                        focus="role",
                    )
                if "404" in text:
                    return self._oic_message(
                        "OIC MCP returned 404 — wrong project ID, or 'Enable MCP server' "
                        "is off on the project.",
                        focus="project",
                    )

        # ADB — auth endpoint or MCP endpoint failures
        if ADB_ENABLED:
            if "/adb/auth/" in text_lc and "401" in text:
                return self._adb_message(
                    "ADB rejected the password-grant credentials (401)."
                )
            if "/adb/mcp/" in text_lc and any(s in text for s in ("401", "403")):
                return self._adb_message(
                    "ADB MCP rejected the access token. Token may have expired between "
                    "fetch and use, or the user lacks Select AI execute privileges."
                )
            if "/adb/" in text_lc and "404" in text:
                return self._adb_message(
                    "ADB endpoint returned 404 — likely wrong OCID or region."
                )

        return None

    @staticmethod
    def _oac_jwt_message(reason: str) -> str:
        return (
            "🔑 OAC JWT authentication failed.\n\n"
            f"Reason: {reason}\n\n"
            "How to fix:\n"
            "  1. Verify the Confidential App in your OCI Identity Domain:\n"
            "       • Status is Active\n"
            "       • 'JWT Assertion' grant type is enabled\n"
            "       • Your X.509 cert (oac-cert.pem) is registered as a\n"
            "         Trusted Client on the app\n"
            "  2. Verify integrations.oac in config.yaml:\n"
            "       • client_id matches the Confidential App\n"
            "       • idcs_host matches your OCI Identity Domain\n"
            "       • service_user_email is an Active OAC user with the\n"
            "         permissions you want the agent to inherit\n"
            "  3. Verify the private key file (default: oac-jwt-key.txt) is\n"
            "     uploaded next to agent.py and matches the cert registered\n"
            "     in IDCS.\n"
            "  4. Use mint-and-exchange.py locally to test the full chain\n"
            "     before redeploying the agent.\n"
            "  5. Redeploy after fixing.\n"
        )

    @staticmethod
    def _oic_message(reason: str, focus: str) -> str:
        steps = {
            "role": (
                "  1. OCI Console → Identity → Domains → your domain →\n"
                "     Oracle Cloud Services → your OIC instance → Application Roles\n"
                "  2. Open ServiceInvoker → Assigned applications → assign the\n"
                "     Confidential App referenced in config.yaml (integrations.oic.client_id)\n"
                "  3. Save. The role propagates within ~60 seconds.\n"
                "  4. Re-send your message — no redeploy needed.\n"
                "\n"
                "Note: ServiceAdministrator alone is NOT enough — MCP requires ServiceInvoker."
            ),
            "scope": (
                "  1. Open the Confidential App in your OCI Identity Domain\n"
                "  2. OAuth Configuration → Token issuance policy → Resources\n"
                "  3. Copy BOTH scope values verbatim (urn:opc:resource:consumer::all\n"
                "     AND /ic/api/) — they use the IDCS-registered hash hostname,\n"
                "     NOT the user-facing design.* hostname\n"
                "  4. Paste them space-separated into integrations.oic.scope in config.yaml\n"
                "  5. Redeploy.\n"
            ),
            "client": (
                "  1. Verify integrations.oic.client_id and client_secret in config.yaml\n"
                "     match an Active Confidential Application in your OCI Identity Domain\n"
                "  2. App detail page → status badge should say 'Active'. If 'Inactive':\n"
                "     Actions → Activate.\n"
                "  3. Allowed grant types must include 'Client Credentials'.\n"
                "  4. Redeploy after fixing.\n"
            ),
            "project": (
                "  1. OIC Designer → open the project → pencil/Edit (top right) →\n"
                "     ☑ Enable MCP server → Save\n"
                "  2. Verify integrations.oic.mcp_url in config.yaml matches the URL shown\n"
                "     after saving. Canonical pattern:\n"
                "     https://<host>/mcp-server/v1/projects/<projectId>/mcp\n"
                "  3. Redeploy.\n"
            ),
        }[focus]
        return (
            "🔑 OIC authentication / authorization failed.\n\n"
            f"Reason: {reason}\n\n"
            "How to fix:\n"
            f"{steps}"
        )

    @staticmethod
    def _adb_message(reason: str) -> str:
        return (
            "🔑 ADB authentication failed.\n\n"
            f"Reason: {reason}\n\n"
            "How to fix:\n"
            "  1. Verify integrations.adb in config.yaml: ocid, region, user, password\n"
            "  2. Sanity check by logging into the same ADB via SQL Developer Web with\n"
            "     the same user/password\n"
            "  3. Confirm the user has Select AI privileges:\n"
            "       GRANT EXECUTE ON DBMS_CLOUD_AI TO <user>;\n"
            "  4. Redeploy after fixing config.yaml\n"
        )

    # ── Invoke (per user message) ───────────────────────────────
    async def invoke(self, user_query: str, **kwargs):
        """Called by AIDP for every chat message. The high-level flow:

        1. Bail out early if setup() captured an error (config or auth
           setup failed) — return that error as the chat reply.
        2. Run pre_invoke_setup() to configure auth context for OCI
           Generative AI.
        2.5. Refresh the OAC bearer if it's expired or about to expire
           (5-minute token TTL — handles AIDP suspend-then-wake gaps).
        3. Lazy-load MCP tools the very first time we're invoked.
        4. Heal any orphan tool_calls left in conversation state by
           previously-failed tool executions.
        5. Run the actual react-agent graph.
        6. If the graph errored with something recoverable (auth/network),
           rebuild MCP, heal orphans, and retry once.

        There is NO scheduled token refresh. Every Oracle service we hit
        can be re-authenticated on demand from cached config + private
        key, with no rotating refresh chain to keep alive. The proactive
        check in step 2.5 handles the common case (AIDP idle for hours,
        cached OAC token expired); step 6 catches the rest."""

        # ─── Step 1: surface setup-time errors ─────────────────────
        if self._self_stage_error:
            return {"messages": [{"role": "ai", "content":
                f"Self-stage failed at setup:\n{self._self_stage_error}\n\n"
                f"Runtime diagnostics:\n{self._runtime_diagnostics()}"
            }]}

        # ─── Step 2: AIDP auth context for OCI Gen AI ──────────────
        # pre_invoke_setup wires the OCI signer into the request thread so
        # the LLM's HTTP calls get signed correctly. If it errors we soldier
        # on with an empty config — chat may still partially work.
        try:
            config = pre_invoke_setup(**kwargs)
        except Exception as e:
            logger.warning("pre_invoke_setup failed, using empty config: %s", e)
            config = {}

        # Snapshot of what pre_invoke_setup returned, sanitized for the
        # debug block. We care primarily about whether configurable.thread_id
        # is present — without it langgraph's checkpointer can't persist
        # conversation state and aget_state() raises ValueError.
        try:
            cfg_top_keys = sorted((config or {}).keys())
            cfg_inner_keys = sorted(((config or {}).get("configurable") or {}).keys())
            has_thread = "thread_id" in cfg_inner_keys
            self._last_config_keys = (
                f"top={cfg_top_keys}; configurable={cfg_inner_keys}; "
                f"thread_id={'✓' if has_thread else '✗'}"
            )
        except Exception:
            self._last_config_keys = "(unable to introspect config)"

        # asyncio.Lock has to be created INSIDE a running event loop.
        # That's why we lazy-init here on the first invoke instead of
        # in __init__ (which runs at module import time, no loop).
        if self._load_lock is None:
            self._load_lock = asyncio.Lock()

        # ─── Step 2.5: proactive token freshness for all enabled services ──
        # Handles OAC's 5-minute TTL and ADB/OIC's 1-hour TTL uniformly.
        # Cheap when nothing needs refreshing (a couple of comparisons).
        # Catches AIDP-idle-suspend-then-wake gaps before they cause 401s.
        try:
            await self._ensure_fresh_tokens()
        except Exception as e:
            logger.warning("Token freshness check failed (continuing): %s", e)

        # ─── Step 3: lazy first-load of MCP tools ──────────────────
        # We don't load MCP at setup() time because setup is sync and MCP
        # loading is async. So the first chat message pays a one-time cost
        # of opening sessions and discovering tools. Subsequent messages
        # use the cached graph until something forces a rebuild (token
        # expiry caught by reactive retry in Step 6).
        # Double-checked locking: outer check is the fast path, the inner
        # check inside the lock prevents two concurrent first-invokes from
        # both running the load.
        if not self._tools_loaded:
            async with self._load_lock:
                if not self._tools_loaded:
                    try:
                        await self._load_all_mcp_tools()
                    except Exception as e:
                        logger.error("MCP tool load failed: %s", e, exc_info=True)
                        friendly = self._friendly_auth_message(e)
                        if friendly:
                            return {"messages": [{"role": "ai", "content": friendly}]}
                        return {"messages": [{"role": "ai", "content":
                            f"MCP setup error: {type(e).__name__}: {e}\n\n"
                            f"Runtime diagnostics:\n{self._runtime_diagnostics()}\n\n"
                            f"Stack trace (last 800 chars):\n{traceback.format_exc()[-800:]}"
                        }]}

        # ─── Step 4: heal orphan tool_calls from prior failures ────
        try:
            await self._heal_orphan_tool_calls(config)
        except Exception as e:
            logger.warning("Orphan healing failed (continuing): %s", e)

        # ─── Step 4.5: trim oldest history if over the configured budget ──
        # Keeps the LLM call within the context window even on long sessions.
        # No-op when llm.max_context_tokens is unset (= 0) in config.yaml.
        # The method populates self._last_trim_* with stats either way, so
        # the optional DEBUG block can show what happened (or didn't).
        try:
            await self._trim_history(config)
        except Exception as e:
            logger.warning("History trim failed (continuing): %s", e)

        # ─── Step 5: run the react-agent graph ─────────────────────
        # The graph handles the LLM ↔ tool-call ↔ LLM dance. It runs to
        # completion (final answer or error) before returning here.
        user_message = HumanMessage(content=user_query)
        messages = {"messages": [dict(user_message)]}

        try:
            result = await self.graph.ainvoke(messages, config=config)
            if DEBUG_MODE:
                self._inject_debug(result, "OK", None)
            return result
        except Exception as e:
            # ─── Step 6: reactive recovery ────────────────────────
            # If the failure looks transient (auth/network), try once more
            # after rebuilding MCP. This catches the common case where a
            # token expired in flight or a TCP socket got dropped.
            if _is_recoverable_error(e):
                logger.warning(
                    "Recoverable error mid-invoke (%s); rebuilding MCP and retrying",
                    type(e).__name__,
                )
                try:
                    # Rebuild under the lock so concurrent invokes don't
                    # both kick off a rebuild.
                    async with self._load_lock:
                        if OAC_ENABLED:
                            await self._refresh_oac_token_and_rebuild()
                        else:
                            await self._load_all_mcp_tools()
                    # The failed call may have left an orphan tool_call in
                    # conversation state — heal it before retrying or the
                    # LLM provider will reject the request.
                    try:
                        await self._heal_orphan_tool_calls(config)
                    except Exception as heal_err:
                        logger.warning("Orphan healing on retry failed: %s", heal_err)
                    result = await self.graph.ainvoke(messages, config=config)
                    if DEBUG_MODE:
                        self._inject_debug(result, "OK (after retry)", None)
                    return result
                except Exception as e2:
                    # Retry also failed. Surface a friendly auth message if
                    # we can classify it as an identifiable auth issue.
                    friendly = self._friendly_auth_message(e2)
                    if friendly:
                        return {"messages": [{"role": "ai", "content": friendly}]}
                    # If the retry hit INVALID_CHAT_HISTORY (orphan tool_call
                    # we couldn't heal), do a hard state wipe and try ONE more
                    # time. Loses prior conversation memory but recovers
                    # automatically.
                    if _is_invalid_history(e2):
                        logger.warning(
                            "Retry hit INVALID_CHAT_HISTORY — hard-clearing state and trying once more"
                        )
                        try:
                            await self._hard_clear_history(config)
                            return await self.graph.ainvoke(messages, config=config)
                        except Exception as e3:
                            logger.error("Hard-clear retry also failed: %s", e3, exc_info=True)
                    logger.error("Retry after rebuild failed: %s", e2, exc_info=True)
                    return {"messages": [{"role": "ai", "content":
                        self._try_again_message()
                    }]}

            # INVALID_CHAT_HISTORY caught directly (no recoverable error
            # preceded it). Same hard-clear + retry path.
            if _is_invalid_history(e):
                logger.warning(
                    "INVALID_CHAT_HISTORY mid-invoke — hard-clearing state and retrying"
                )
                try:
                    await self._hard_clear_history(config)
                    return await self.graph.ainvoke(messages, config=config)
                except Exception as e2:
                    logger.error("Hard-clear retry failed: %s", e2, exc_info=True)
                return {"messages": [{"role": "ai", "content":
                    self._try_again_message()
                }]}

            # Non-recoverable, unclassified error path: log full detail for
            # ops, but only show the user a friendly retry message.
            logger.error("invoke error: %s", e, exc_info=True)
            return {"messages": [{"role": "ai", "content":
                self._try_again_message()
            }]}

    def _inject_debug(
        self,
        result,
        rebuild_status: str,
        rebuild_error: str | None,
    ) -> None:
        """Prepend a compact debug block to the agent's last AI message.

        Gated by the `DEBUG_MODE` constant at the top of this module.
        When `DEBUG_MODE = False`, this method is never called.

        Shows: turn timestamp, rebuild status, tool catalog the agent
        had bound this turn, OAC token TTL if enabled, MCP graph age,
        history trim stats. Useful when AIDP logs aren't accessible.
        Mutates `result` in place — does not return."""
        try:
            ts = time.strftime("%H:%M:%S", time.gmtime())
            tools_csv = ", ".join(self._last_tool_names) or "<none>"
            lines = [
                "🔧 DEBUG",
                f"  • turn at {ts}Z",
                f"  • rebuild: {rebuild_status}",
                f"  • tools ({len(self._last_tool_names)}): {tools_csv}",
            ]
            if rebuild_error:
                lines.append(f"  • rebuild error: {rebuild_error}")
            if self._mcp_built_epoch:
                age = int(time.time() - self._mcp_built_epoch)
                lines.append(f"  • MCP graph age: {age}s")
            if OAC_ENABLED:
                left = self._oac_token_seconds_remaining()
                lines.append(f"  • OAC token valid for: {left}s")
            # langgraph config — critical for state persistence. If
            # thread_id is ✗, the checkpointer is effectively disabled
            # and the agent runs stateless across turns.
            if self._last_config_keys:
                lines.append(f"  • config: {self._last_config_keys}")
            # History trim — show even when LLM_MAX_CONTEXT_TOKENS == 0
            # so the user can confirm trimming IS off vs. silently broken.
            lines.append(
                f"  • history: {self._last_trim_msg_count} msg(s) "
                f"in state ≈ {self._last_trim_msg_tokens_est} tok"
            )
            lines.append(
                f"  • trim: {self._last_trim_status or 'n/a'} "
                f"(max={LLM_MAX_CONTEXT_TOKENS}, "
                f"reserve={LLM_RESPONSE_RESERVE_TOKENS}, "
                f"tools≈{self._tools_token_estimate})"
            )
            block = "\n".join(lines) + "\n\n────────────────\n\n"

            if not isinstance(result, dict):
                return
            msgs = result.get("messages") or []
            if not msgs:
                return
            last = msgs[-1]
            # langgraph returns BaseMessage objects; some chat UIs use dicts.
            current = getattr(last, "content", None)
            if current is not None:
                last.content = block + (current or "")
                return
            if isinstance(last, dict):
                last["content"] = block + (last.get("content") or "")
        except Exception as e:
            # Never let debug instrumentation break the response.
            logger.warning("Debug injection failed (ignored): %s", e)

    @staticmethod
    def _try_again_message() -> str:
        """Friendly message shown to the user when the agent couldn't complete
        the request after all internal retries. Hides the technical details
        (logged separately for ops) and tells the user to try again."""
        return (
            "I ran into a problem completing that request and couldn't recover "
            "from it automatically. Please try sending your message again — if "
            "the issue persists for a few minutes, the connected services "
            "(database, analytics, integrations) may be having trouble. The "
            "agent's logs have full details for diagnostics."
        )
