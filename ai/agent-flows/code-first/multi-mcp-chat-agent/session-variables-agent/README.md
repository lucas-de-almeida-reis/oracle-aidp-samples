# Multi-MCP Chat Agent — session-variable auth (simplified)

A simplified sibling of the [parent multi-MCP agent](../agent.py). Same job —
one chat agent fronting **ADB Select AI**, **OAC**, and **OIC** MCP servers —
but built around the product's **intended authentication model** and stripped
of the heavy state-management machinery.

## How it differs from `../agent.py`

| | Parent (`../agent.py`) | This agent |
|---|---|---|
| Token handling | **Mints** bearers itself (ADB password grant, OAC JWT assertion, OIC client_credentials) | **Never mints** — bearer arrives as a session variable |
| Secrets in AIDP | DB password, client secret, JWT private key | **None** |
| Tool libs | `MultiServerMCPClient` + `create_react_agent` | aidputils `build_structured_tools_from_allowed_mcp_tools` + `create_agent` |
| Tool discovery | live, lazy on first invoke | live, lazy on first invoke (shared catalogue) |
| Auth resolution | bearer baked into headers at build | `{{sessionvariables…}}` resolved **per call** (per user) |
| State mgmt | trim / heal / hard-clear / overflow / `/reset` | **removed** (kept minimal on purpose) |

## The authentication model

The agent **does not generate tokens**. For each MCP server it declares a
bearer via a session-variable placeholder:

```python
auth = {"authType": "BEARER_TOKEN",
        "token": "{{sessionvariables.cred.mcp.adb.bearer}}"}
```

aidputils' `BearerTokenAuthStrategy` resolves that placeholder **on every tool
call (and on discovery)** from the current request's session context, which
`pre_tool_setup(**kwargs)` populates from `kwargs["session_variables"]`.

- **DEV** — you generate a bearer by hand and store it in the Agent Studio
  **Variables** tab (the credential store). It is delivered as a session
  variable on each request.
- **PROD** — the **calling application** passes the bearer per request as a
  session variable. Token generation stays **outside** AIDP.

There is **no dev-vs-prod branch** in the code: both put the value in the same
session variable. If it's absent, resolution raises and the agent returns a
clear "no bearer available" message.

### Per-user bearer, shared tool list

The MCP client is cached per server, but the `Authorization` header is rebuilt
on every session from the per-request context. So:

- the **tool catalogue** is discovered once and shared across users;
- each **tool call** uses **that one user's** bearer — never another user's.

## Session variables to declare

Declare one credential session variable per enabled integration (Agent Studio
→ Variables tab, or the `session_config` block the agent builds at import):

| Integration | Session variable (default) |
|---|---|
| ADB | `sessionvariables.cred.mcp.adb.bearer` |
| OAC | `sessionvariables.cred.mcp.oac.bearer` |
| OIC | `sessionvariables.cred.mcp.oic.bearer` |

Mark them `isSystem: true`, `shouldLog: false` (secret; value never logged).
The names are configurable per integration in `config.yaml`.

## Configuration

Copy `config.sample.yaml` → `config.yaml`. Because no tokens are minted, the
config is small — per integration only:

- `enabled`
- `server_name` — logical name for the MCP server
- `endpoint` — the MCP URL
- `transport` — usually `streamable_http`
- `bearer_session_variable` — the session-variable name (no `{{ }}`)
- `headers` — optional (OIC needs `Accept: application/json, text/event-stream`)

## Local testing (outside AIDP)

There are no session variables locally, so `agent.py`'s `_main()` fakes them
exactly as the runtime delivers them — under the `session_variables` kwarg:

```python
await agent.invoke(
    "your question",
    session_variables={
        "sessionvariables.cred.mcp.adb.bearer": {"value": "<a real bearer>"},
    },
    thread_id="local-test-thread",
)
```

## Context management

Ported from [`../agent.py`](../agent.py) to keep conversations stable against
long-context tool-use degradation (worsened by OCI generic-provider schema
flattening):

- **`MAX_TURNS_KEPT`** (constant in `agent.py`, default 5) — hard cap on
  recent user turns. The primary, always-on lever.
- **`max_context_tokens`** (config, default 0/off) — optional token-budget
  trim, sized from the real `prompt_tokens` OCI reports each turn.
- **orphan tool_call healing** — drops `AIMessage(tool_calls)` left without a
  matching `ToolMessage` (e.g. AIDP suspended mid-call).
- **`/reset`** — wipes conversation state on demand.
- **context-overflow / INVALID_CHAT_HISTORY recovery**.

State ops use the compiled-graph API (`agent.aget_state` / `aupdate_state`)
and degrade gracefully to no-ops if a runtime doesn't expose them.

## What is still NOT here (vs the parent)

The parent's **token-minting + reactive MCP-rebuild retry** is intentionally
absent: this agent never mints tokens, and the bearer is resolved per call
from the session variable, so a 401 means "the caller must supply a valid
bearer", not "rebuild and retry".
