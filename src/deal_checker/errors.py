"""Turns exceptions into actionable messages: what went wrong and how to fix it.

Used by the CLI (`src/main.py`). Transient model errors (429/5xx) are retried
automatically first (see MODEL_CONFIG in agent.py); what reaches this module
has already survived the retries or is not retryable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from google.auth import exceptions as auth_exceptions
from google.genai import errors as genai_errors

ENV_FILE = ".env"


@dataclass(frozen=True)
class GuidedError:
    """A user-facing explanation of a failure."""

    title: str  # what went wrong, one line
    hint: str  # what to do about it (concrete commands where possible)
    retryable: bool = False  # True if simply trying again later may work

    def render(self) -> str:
        retry = "\n  (This is usually temporary: trying again in a minute may work.)" if self.retryable else ""
        return f"Error: {self.title}\n  Fix: {self.hint}{retry}"


def _message(exc: BaseException) -> str:
    return str(getattr(exc, "message", None) or exc)


def _api_error(exc: genai_errors.APIError) -> GuidedError:
    code, msg = exc.code, _message(exc)
    lower = msg.lower()
    agent_engine = "reasoningengines" in lower

    if code == 400 and ("api key" in lower or "api_key_invalid" in lower):
        return GuidedError(
            "The Gemini API key was rejected.",
            f"Check GOOGLE_API_KEY in {ENV_FILE} (no quotes or spaces), or create a new key at "
            "https://aistudio.google.com/apikey.",
        )
    if code in (401, 403):
        if "service_disabled" in lower or "has not been used" in lower or "is disabled" in lower:
            return GuidedError(
                "A required Google Cloud API is not enabled for this project.",
                "Run: GOOGLE_API_USE_CLIENT_CERTIFICATE=false bash scripts/enable_apis.sh",
            )
        return GuidedError(
            "Permission denied by Google Cloud.",
            "Refresh your credentials (gcloud auth application-default login) and make sure your account "
            f"has the 'Vertex AI User' role on project {os.getenv('GOOGLE_CLOUD_PROJECT', '<project>')}.",
        )
    if code == 404:
        if agent_engine:
            return GuidedError(
                "The Agent Engine (sessions / Memory Bank) or the session was not found.",
                f"Check SESSION_SERVICE_URI and MEMORY_SERVICE_URI in {ENV_FILE}. To create a new Agent Engine "
                "run: uv run python scripts/create_agent_engine.py. If --resume picked a deleted session, "
                "run without --resume.",
            )
        return GuidedError(
            "The model or resource was not found.",
            "Check CHECK_MODEL / FOLLOWUP_MODEL in src/deal_checker/agent.py are available model names.",
        )
    if code == 429:
        return GuidedError(
            "Rate limit or quota exceeded (still failing after automatic retries).",
            "Wait a minute and try again. If it keeps happening, check quotas in Cloud Console: "
            "IAM & Admin > Quotas (Generative Language / Vertex AI).",
            retryable=True,
        )
    if code == 400:
        return GuidedError(
            f"The request was rejected as invalid: {msg[:200]}",
            "If you sent images, make sure they open correctly and are JPEG, PNG or WebP screenshots.",
        )
    if isinstance(exc, genai_errors.ServerError) or (code and code >= 500):
        return GuidedError(
            f"Google's service returned an error ({code}) after automatic retries.",
            "Try again in a few minutes. Check https://status.cloud.google.com if it persists.",
            retryable=True,
        )
    return GuidedError(f"Google API error {code}: {msg[:200]}", "Re-run with --debug for the full error.")


def explain(exc: BaseException) -> GuidedError:
    """Maps an exception to a GuidedError (with a generic fallback)."""
    if isinstance(exc, genai_errors.APIError):
        return _api_error(exc)
    if isinstance(exc, (auth_exceptions.MutualTLSChannelError, auth_exceptions.ClientCertError)):
        return GuidedError(
            "The client-certificate (mTLS) helper failed on this machine.",
            f"Add GOOGLE_API_USE_CLIENT_CERTIFICATE=false to {ENV_FILE}.",
        )
    if isinstance(exc, auth_exceptions.DefaultCredentialsError):
        return GuidedError(
            "No Google Cloud credentials found (needed for sessions, Memory Bank and Model Armor).",
            "Run: gcloud auth application-default login",
        )
    if isinstance(exc, auth_exceptions.RefreshError):
        return GuidedError(
            "Your Google Cloud credentials have expired or were revoked.",
            "Run: gcloud auth application-default login",
        )
    try:
        from google.api_core import exceptions as core_exceptions
    except ImportError:  # pragma: no cover - api_core ships with the Cloud clients
        core_exceptions = None
    if core_exceptions and isinstance(exc, core_exceptions.PermissionDenied):
        return GuidedError(
            f"Permission denied by Google Cloud: {_message(exc)[:200]}",
            "Run scripts/enable_apis.sh and check your account's IAM roles on the project.",
        )
    if core_exceptions and isinstance(exc, core_exceptions.NotFound):
        return GuidedError(
            f"Google Cloud resource not found: {_message(exc)[:200]}",
            f"Check the resource names in {ENV_FILE} (Agent Engine, Model Armor templates).",
        )
    if isinstance(exc, (ConnectionError, TimeoutError)) or type(exc).__name__ in {"ConnectError", "ConnectTimeout", "ReadTimeout"}:
        return GuidedError(
            "Could not reach Google's servers.",
            "Check your internet connection / proxy and try again.",
            retryable=True,
        )
    if isinstance(exc, ValueError) and "SERVICE_URI" in str(exc):
        return GuidedError(
            str(exc),
            "Use sqlite:///<path>, memory:// or agentengine://projects/<number>/locations/<region>/"
            f"reasoningEngines/<id> in {ENV_FILE}, or remove the variable to use local storage.",
        )
    return GuidedError(
        f"Unexpected error ({type(exc).__name__}): {_message(exc)[:200]}",
        "Re-run with --debug to see the full traceback; structured logs have the event 'run.error'.",
    )


def check_config() -> list[GuidedError]:
    """Pre-flight checks of the environment, before any API call is made."""
    problems: list[GuidedError] = []
    use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in {"1", "true"}
    if not use_vertex and not os.getenv("GOOGLE_API_KEY"):
        problems.append(GuidedError(
            "No Gemini credentials configured.",
            f"Set GOOGLE_API_KEY in {ENV_FILE} (https://aistudio.google.com/apikey), or set "
            "GOOGLE_GENAI_USE_VERTEXAI=true with GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION.",
        ))
    if use_vertex and not os.getenv("GOOGLE_CLOUD_PROJECT"):
        problems.append(GuidedError(
            "GOOGLE_GENAI_USE_VERTEXAI=true but GOOGLE_CLOUD_PROJECT is not set.",
            f"Add GOOGLE_CLOUD_PROJECT=<project-id> to {ENV_FILE}.",
        ))
    for var in ("SESSION_SERVICE_URI", "MEMORY_SERVICE_URI"):
        uri = os.getenv(var, "")
        if uri.startswith("agentengine://") and len(uri) <= len("agentengine://"):
            problems.append(GuidedError(
                f"{var} has no Agent Engine ID.",
                f"Use {var}=agentengine://projects/<number>/locations/<region>/reasoningEngines/<id>, or "
                "remove it to use local storage.",
            ))
    return problems
