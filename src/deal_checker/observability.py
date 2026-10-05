"""Structured JSON logging, distributed tracing and the observability plugin.

Configuration (environment variables):
  LOG_FORMAT      json (default) | text    - format written to stderr
  LOG_LEVEL       INFO (default) | DEBUG | WARNING ...
  LOG_TO_CLOUD    false (default) | true   - also send logs to Google Cloud Logging
                                             (log name "deal-checker", project from
                                             GOOGLE_CLOUD_PROJECT / ADC)
  LOG_CONTENT     false (default) | true   - include user/model text in logs and spans
                                             (local debugging only; off by default to
                                             keep personal data out of telemetry)
  TRACE_TO_CLOUD  false (default) | true   - export ADK's OpenTelemetry spans to Cloud
                                             Trace (CLI). For `adk web`, which installs
                                             its own tracer provider, use
                                             `adk web --otel_to_cloud` instead.

Every record is a JSON object with an `event` name plus context fields
(`invocation_id`, `session_id`, `user`, `agent`), so runs can be filtered and
correlated in Cloud Logging, e.g. `jsonPayload.event="model.outcome"`.
When tracing is on, each log entry also carries the active trace/span ID
(Cloud Logging links it to the trace) and each event is added to the active
span, so the trace waterfall shows guardrail and routing decisions inline.

`ObservabilityPlugin` records intent before and outcome after each step:
  invocation.start / invocation.end   - one user turn
  agent.start      / agent.end        - each agent in the graph
  model.intent     / model.outcome    - each Gemini call (incl. the Google Search
                                        queries the model actually ran, tokens, latency)
  tool.intent      / tool.outcome     - each tool call (argument names only, status,
                                        result count, latency)
  model.error / agent.error / run.error / tool.error
Guardrail decisions are logged where they are made as `guardrail.*` events.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

LOGGER_NAME = "deal_checker"
CLOUD_LOG_NAME = "deal-checker"
MAX_CONTENT_CHARS = 500

logger = logging.getLogger(LOGGER_NAME)


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes"}


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


class _TextFormatter(logging.Formatter):
    """Human-readable alternative: `time LEVEL event key=value ...`."""

    def format(self, record: logging.LogRecord) -> str:
        fields = getattr(record, "json_fields", {}) or {}
        pairs = " ".join(
            f"{k}={json.dumps(v, default=str)}" for k, v in fields.items() if k not in ("event", "timestamp")
        )
        ts = time.strftime("%H:%M:%S", time.localtime(record.created))
        return f"{ts} {record.levelname:<7} {record.getMessage()} {pairs}".rstrip()


def setup_logging() -> None:
    """Configures the `deal_checker` logger tree once (idempotent).

    Only the app's own logger is configured (not the root logger), so ADK and
    library logging are unaffected. Called at agent import time so `adk web`
    and the CLI behave the same.
    """
    if getattr(logger, "_deal_checker_configured", False):
        return
    logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    logger.propagate = False

    if os.getenv("LOG_FORMAT", "json").lower() == "text":
        handler: logging.Handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_TextFormatter())
    else:
        from google.cloud.logging.handlers import StructuredLogHandler

        handler = StructuredLogHandler(stream=sys.stderr, project_id=os.getenv("GOOGLE_CLOUD_PROJECT"))
    logger.addHandler(handler)

    if _env_flag("LOG_TO_CLOUD"):
        try:
            import google.cloud.logging
            from google.cloud.logging import Resource
            from google.cloud.logging.handlers import CloudLoggingHandler

            client = google.cloud.logging.Client(project=os.getenv("GOOGLE_CLOUD_PROJECT"))
            resource = Resource(type="global", labels={"project_id": client.project})
            logger.addHandler(CloudLoggingHandler(
                client, name=CLOUD_LOG_NAME, resource=resource, labels={"app": "deal_checker"}
            ))
        except Exception as exc:  # Logging must never break the agent.
            logger.warning("Cloud Logging disabled: %s", exc)

    logger._deal_checker_configured = True  # type: ignore[attr-defined]


def setup_tracing() -> None:
    """Exports ADK's OpenTelemetry spans to Cloud Trace when TRACE_TO_CLOUD=true.

    ADK already creates spans (invocation -> agent_run -> call_llm/execute_tool);
    this only installs a tracer provider that sends them to Cloud Trace via the
    Telemetry API (telemetry.googleapis.com), using ADK's own GCP exporter.
    """
    os.environ.setdefault("ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS", "true" if _env_flag("LOG_CONTENT") else "false")
    if not _env_flag("TRACE_TO_CLOUD"):
        return

    from opentelemetry import trace

    if not isinstance(trace.get_tracer_provider(), trace.ProxyTracerProvider):
        # A provider is already installed (e.g. by `adk web`); never override it.
        logger.info(
            "Tracer provider already configured; for Cloud Trace with adk web use `adk web --otel_to_cloud`."
        )
        return
    try:
        import google.auth
        from google.adk.telemetry.google_cloud import get_gcp_exporters, get_gcp_resource
        from google.adk.telemetry.setup import maybe_set_otel_providers

        os.environ.setdefault("OTEL_SERVICE_NAME", CLOUD_LOG_NAME)
        credentials, default_project = google.auth.default()
        project = os.getenv("GOOGLE_CLOUD_PROJECT") or default_project
        # BatchSpanProcessor exports in the background; the SDK flushes it at exit.
        maybe_set_otel_providers(
            otel_hooks_to_setup=[get_gcp_exporters(enable_cloud_tracing=True, google_auth=(credentials, project))],
            otel_resource=get_gcp_resource(project),
        )
    except Exception as exc:  # Tracing must never break the agent.
        logger.warning("Cloud Trace disabled: %s", exc)


def _span_attribute(value: Any) -> Any:
    """Converts a log field to a valid OpenTelemetry attribute value."""
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
        return list(value)
    return json.dumps(value, default=str)


def _add_span_event(event: str, fields: dict[str, Any]) -> None:
    """Mirrors a log event onto the active span (shown inline in Cloud Trace)."""
    from opentelemetry import trace

    span = trace.get_current_span()
    if span.is_recording():
        span.add_event(event, {k: _span_attribute(v) for k, v in fields.items() if v is not None})


def _hash_user(user_id: str | None) -> str | None:
    """Pseudonymous user id: stable for correlation, not reversible to the raw id."""
    return hashlib.sha256(user_id.encode()).hexdigest()[:12] if user_id else None


def _context_fields(ctx: InvocationContext | CallbackContext | None) -> dict[str, Any]:
    if ctx is None:
        return {}
    if isinstance(ctx, InvocationContext):
        return {
            "invocation_id": ctx.invocation_id,
            "session_id": ctx.session.id,
            "user": _hash_user(ctx.user_id),
        }
    inv = getattr(ctx, "_invocation_context", None)
    if inv is None:
        return {}  # e.g. a test double
    return {
        "invocation_id": ctx.invocation_id,
        "session_id": inv.session.id,
        "user": _hash_user(inv.user_id),
        "agent": ctx.agent_name,
    }


def log_event(
    event: str,
    ctx: InvocationContext | CallbackContext | None = None,
    *,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    """Logs one structured event and records it on the active trace span.

    `fields` must not contain raw personal data. Trace/span IDs are attached to
    the log entry automatically by google-cloud-logging from the active span.
    """
    payload = {
        "event": event,
        # RFC 3339; Cloud Logging uses it as the entry time when ingesting stdout/stderr.
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        **_context_fields(ctx),
        **{k: v for k, v in fields.items() if v is not None},
    }
    logger.log(level, event, extra={"json_fields": payload}, stacklevel=2)
    _add_span_event(event, fields)


# ---------------------------------------------------------------------------
# Helpers to summarize requests/responses without logging their content
# ---------------------------------------------------------------------------


def _content_stats(content: types.Content | None) -> dict[str, Any]:
    parts = (content.parts or []) if content else []
    text = "".join(p.text or "" for p in parts if p.text and not p.thought)
    stats: dict[str, Any] = {
        "text_chars": len(text),
        "images": sum(1 for p in parts if p.inline_data and (p.inline_data.mime_type or "").startswith("image/")),
    }
    if _env_flag("LOG_CONTENT") and text:
        stats["text"] = text[:MAX_CONTENT_CHARS]
    return stats


def _tool_names(llm_request: LlmRequest) -> list[str]:
    names: list[str] = []
    for tool in (llm_request.config.tools if llm_request.config and llm_request.config.tools else []):
        if getattr(tool, "google_search", None) is not None:
            names.append("google_search")
        for decl in getattr(tool, "function_declarations", None) or []:
            names.append(decl.name)
    return names


def _usage(llm_response: LlmResponse) -> dict[str, Any]:
    usage = llm_response.usage_metadata
    if not usage:
        return {}
    return {
        "prompt_tokens": usage.prompt_token_count,
        "output_tokens": usage.candidates_token_count,
        "thinking_tokens": usage.thoughts_token_count,
        "total_tokens": usage.total_token_count,
    }


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------


class ObservabilityPlugin(BasePlugin):
    """Logs intent (before) and outcome (after) for runs, agents and model calls.

    Register it first so it observes every step, including runs that a later
    guardrail plugin halts. Never modifies or short-circuits anything.
    """

    def __init__(self) -> None:
        super().__init__(name="deal_checker_observability")
        self._started: dict[tuple[str, ...], float] = {}
        self._run_tokens: dict[str, int] = {}

    def _start(self, *key: str) -> None:
        self._started[key] = time.monotonic()

    def _elapsed_ms(self, *key: str) -> int | None:
        start = self._started.pop(key, None)
        return round((time.monotonic() - start) * 1000) if start is not None else None

    # -- run ----------------------------------------------------------------

    async def before_run_callback(self, *, invocation_context: InvocationContext) -> types.Content | None:
        self._start("run", invocation_context.invocation_id)
        self._run_tokens[invocation_context.invocation_id] = 0
        log_event(
            "invocation.start", invocation_context,
            app=invocation_context.app_name,
            **_content_stats(invocation_context.user_content),
        )
        return None

    async def after_run_callback(self, *, invocation_context: InvocationContext) -> None:
        inv_id = invocation_context.invocation_id
        log_event(
            "invocation.end", invocation_context,
            duration_ms=self._elapsed_ms("run", inv_id),
            total_tokens=self._run_tokens.pop(inv_id, None),
        )

    async def on_run_error_callback(self, *, invocation_context: InvocationContext, error: Exception) -> None:
        inv_id = invocation_context.invocation_id
        self._run_tokens.pop(inv_id, None)
        log_event(
            "run.error", invocation_context, level=logging.ERROR,
            error_type=type(error).__name__, error=str(error)[:MAX_CONTENT_CHARS],
            duration_ms=self._elapsed_ms("run", inv_id),
        )

    # -- agents -------------------------------------------------------------

    async def before_agent_callback(self, *, agent: BaseAgent, callback_context: CallbackContext) -> types.Content | None:
        self._start("agent", callback_context.invocation_id, agent.name)
        log_event("agent.start", callback_context, agent_type=type(agent).__name__)
        return None

    async def after_agent_callback(self, *, agent: BaseAgent, callback_context: CallbackContext) -> types.Content | None:
        log_event(
            "agent.end", callback_context,
            duration_ms=self._elapsed_ms("agent", callback_context.invocation_id, agent.name),
            output_key=getattr(agent, "output_key", None),
        )
        return None

    async def on_agent_error_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext, error: Exception
    ) -> None:
        log_event(
            "agent.error", callback_context, level=logging.ERROR,
            error_type=type(error).__name__, error=str(error)[:MAX_CONTENT_CHARS],
        )

    # -- model calls ----------------------------------------------------------

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> LlmResponse | None:
        self._start("model", callback_context.invocation_id, callback_context.agent_name)
        config = llm_request.config
        log_event(
            "model.intent", callback_context,
            model=llm_request.model,
            tools=_tool_names(llm_request),
            structured_output=bool(config and config.response_schema),
            history_messages=len(llm_request.contents or []),
            **(_content_stats(llm_request.contents[-1]) if llm_request.contents else {}),
        )
        return None

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> LlmResponse | None:
        if llm_response.partial:
            return None  # streaming chunk; the final response is logged
        grounding = llm_response.grounding_metadata
        usage = _usage(llm_response)
        inv_id = callback_context.invocation_id
        if inv_id in self._run_tokens and usage.get("total_tokens"):
            self._run_tokens[inv_id] += usage["total_tokens"]
        log_event(
            "model.outcome", callback_context,
            level=logging.WARNING if llm_response.error_code else logging.INFO,
            duration_ms=self._elapsed_ms("model", inv_id, callback_context.agent_name),
            finish_reason=llm_response.finish_reason.name if llm_response.finish_reason else None,
            error_code=llm_response.error_code,
            search_queries=list(grounding.web_search_queries or []) if grounding else None,
            search_sources=len(grounding.grounding_chunks or []) if grounding else None,
            **usage,
            **_content_stats(llm_response.content),
        )
        return None

    async def on_model_error_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest, error: Exception
    ) -> LlmResponse | None:
        log_event(
            "model.error", callback_context, level=logging.ERROR,
            model=llm_request.model,
            duration_ms=self._elapsed_ms("model", callback_context.invocation_id, callback_context.agent_name),
            error_type=type(error).__name__, error=str(error)[:MAX_CONTENT_CHARS],
        )
        return None

    # -- tool calls -----------------------------------------------------------

    async def before_tool_callback(
        self, *, tool: BaseTool, tool_args: dict[str, Any], tool_context: ToolContext
    ) -> dict | None:
        self._start("tool", tool_context.invocation_id, tool_context.function_call_id or tool.name)
        log_event("tool.intent", tool_context, tool=tool.name, args=sorted(k for k, v in tool_args.items() if v is not None))
        return None

    async def after_tool_callback(
        self, *, tool: BaseTool, tool_args: dict[str, Any], tool_context: ToolContext, result: Any
    ) -> dict | None:
        status = result.get("status") if isinstance(result, dict) else None
        log_event(
            "tool.outcome", tool_context,
            level=logging.WARNING if status == "error" else logging.INFO,
            tool=tool.name,
            duration_ms=self._elapsed_ms("tool", tool_context.invocation_id, tool_context.function_call_id or tool.name),
            status=status,
            error=result.get("error") if status == "error" else None,
            results=len(result["deals"]) if isinstance(result, dict) and isinstance(result.get("deals"), list) else None,
        )
        return None

    async def on_tool_error_callback(
        self, *, tool: BaseTool, tool_args: dict[str, Any], tool_context: ToolContext, error: Exception
    ) -> dict | None:
        log_event(
            "tool.error", tool_context, level=logging.ERROR,
            tool=tool.name,
            duration_ms=self._elapsed_ms("tool", tool_context.invocation_id, tool_context.function_call_id or tool.name),
            error_type=type(error).__name__, error=str(error)[:MAX_CONTENT_CHARS],
        )
        return None
