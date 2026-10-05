"""Guardrails and policy plugins for DealChecker.

Layers (outermost first):
  1. DealCheckerPolicyPlugin   - deterministic input policy (image count/size/type).
  2. DealCheckerModelArmorPlugin - Google Cloud Model Armor screening:
       * user text, once per turn (prompt injection/jailbreak, harmful content)
       * extracted listing text before the validator (indirect injection via the image)
       * user-facing model output (harmful content, sensitive data)
  3. check_verdict             - self-evaluation of the validator's verdict
                                 (structure, consistency, grounding).
  4. SAFETY_CONFIG             - Gemini built-in safety settings on every agent.
"""

from __future__ import annotations

import logging
import os
import re

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.genai import types
from opentelemetry import trace

from .observability import ObservabilityPlugin, log_event

logger = logging.getLogger(__name__)
_tracer = trace.get_tracer("deal_checker.guardrails")

# ---------------------------------------------------------------------------
# Gemini safety settings (applied to every agent)
# ---------------------------------------------------------------------------

SAFETY_CONFIG = types.GenerateContentConfig(
    safety_settings=[
        types.SafetySetting(category=category, threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE)
        for category in (
            types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        )
    ]
)

# ---------------------------------------------------------------------------
# 1. Deterministic input policy
# ---------------------------------------------------------------------------

MAX_IMAGES = 5
MAX_IMAGE_BYTES = 7 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}


def _text(content: types.Content | None) -> str:
    if not content or not content.parts:
        return ""
    return "\n".join(p.text for p in content.parts if p.text and not p.thought)


def _model_message(text: str) -> types.Content:
    return types.Content(role="model", parts=[types.Part(text=text)])


def check_input_policy(content: types.Content | None) -> str | None:
    """Returns a user-facing rejection reason, or None if the message is allowed."""
    if not content or not content.parts:
        return None
    files = [p.inline_data for p in content.parts if p.inline_data]
    if len(files) > MAX_IMAGES:
        return f"Please send at most {MAX_IMAGES} images of a single listing per message (got {len(files)})."
    for blob in files:
        mime = (blob.mime_type or "").lower()
        if mime not in ALLOWED_IMAGE_TYPES:
            return f"Unsupported file type '{mime or 'unknown'}'. Please send JPEG, PNG or WebP images."
        if blob.data and len(blob.data) > MAX_IMAGE_BYTES:
            mb = MAX_IMAGE_BYTES // (1024 * 1024)
            return f"One of the images is larger than {mb} MB. Please send a smaller screenshot."
    return None


class DealCheckerPolicyPlugin(BasePlugin):
    """Rejects messages that violate the input policy before any model runs.

    The offending message is replaced (so oversized/unsupported files are never
    stored in the session) and the run is halted with an explanation.
    """

    def __init__(self) -> None:
        super().__init__(name="deal_checker_policy")
        self._rejections: dict[str, str] = {}

    async def on_user_message_callback(
        self, *, invocation_context: InvocationContext, user_message: types.Content
    ) -> types.Content | None:
        reason = check_input_policy(user_message)
        if not reason:
            return None
        log_event("guardrail.input_policy_rejected", invocation_context, level=logging.WARNING, reason=reason)
        self._rejections[invocation_context.invocation_id] = reason
        return types.Content(role="user", parts=[types.Part(text=f"[Message rejected by input policy: {reason}]")])

    async def before_run_callback(self, *, invocation_context: InvocationContext) -> types.Content | None:
        reason = self._rejections.pop(invocation_context.invocation_id, None)
        return _model_message(reason) if reason else None


# ---------------------------------------------------------------------------
# 2. Model Armor (Google Cloud)
# ---------------------------------------------------------------------------

INPUT_BLOCKED_MESSAGE = (
    "Sorry, I can't help with that request. I can check second-hand listings "
    "and answer questions about them."
)
OUTPUT_BLOCKED_MESSAGE = (
    "Sorry, I couldn't produce a safe answer for this. If the listing contains "
    "personal details (names, phone numbers, emails), try a screenshot without them."
)

SCREEN_REQUEST_AGENTS = {"validator"}
SKIP_RESPONSE_AGENTS = {"extractor_agent", "google_search_agent"}


def build_model_armor_plugin() -> BasePlugin | None:
    """Builds the Model Armor plugin from env vars; None if not configured."""
    prompt_template = os.getenv("MODEL_ARMOR_PROMPT_TEMPLATE")
    response_template = os.getenv("MODEL_ARMOR_RESPONSE_TEMPLATE")
    if not (prompt_template or response_template):
        return None

    from google.adk.integrations.model_armor import ModelArmorConfig, ModelArmorPlugin

    class DealCheckerModelArmorPlugin(ModelArmorPlugin):
        """ModelArmorPlugin tuned for the DealChecker agent graph (see module docstring).

        Each screening call runs in its own `model_armor.<stage>` trace span.
        """

        @staticmethod
        def _record(ctx, span, stage: str, blocked: LlmResponse | None) -> None:
            span.set_attribute("deal_checker.model_armor.blocked", bool(blocked))
            log_event(
                "guardrail.model_armor", ctx,
                level=logging.WARNING if blocked else logging.INFO,
                stage=stage, blocked=bool(blocked),
            )

        async def before_run_callback(self, *, invocation_context: InvocationContext) -> types.Content | None:
            text = _text(invocation_context.user_content)
            if not text or not self._config.prompt_template_name:
                return None
            with _tracer.start_as_current_span("model_armor.user_input") as span:
                try:
                    result = await self._sanitize_user_prompt(text, self._config.prompt_template_name)
                except Exception:  # Outage: block or pass per `block_on_screening_failure`.
                    logger.exception("Model Armor user-input screening failed.")
                    blocked = self._handle_screening_failure(self._config.input_blocked_message)
                else:
                    blocked = self._handle_sanitization_result(
                        result, direction="input", blocked_message=self._config.input_blocked_message
                    )
                self._record(invocation_context, span, "user_input", blocked)
            return blocked.content if blocked else None

        async def before_model_callback(
            self, *, callback_context: CallbackContext, llm_request: LlmRequest
        ) -> LlmResponse | None:
            if callback_context.agent_name not in SCREEN_REQUEST_AGENTS:
                return None  # user text was already screened once in before_run_callback
            with _tracer.start_as_current_span("model_armor.model_request") as span:
                blocked = await super().before_model_callback(
                    callback_context=callback_context, llm_request=llm_request
                )
                self._record(callback_context, span, "model_request", blocked)
            return blocked

        async def after_model_callback(
            self, *, callback_context: CallbackContext, llm_response: LlmResponse
        ) -> LlmResponse | None:
            if callback_context.agent_name in SKIP_RESPONSE_AGENTS:
                return None
            with _tracer.start_as_current_span("model_armor.model_response") as span:
                blocked = await super().after_model_callback(
                    callback_context=callback_context, llm_response=llm_response
                )
                self._record(callback_context, span, "model_response", blocked)
            return blocked

    return DealCheckerModelArmorPlugin(
        config=ModelArmorConfig(
            prompt_template_name=prompt_template,
            response_template_name=response_template,
            input_blocked_message=INPUT_BLOCKED_MESSAGE,
            output_blocked_message=OUTPUT_BLOCKED_MESSAGE,
        )
    )


def build_plugins() -> list[BasePlugin]:
    plugins: list[BasePlugin] = [ObservabilityPlugin(), DealCheckerPolicyPlugin()]
    if model_armor := build_model_armor_plugin():
        plugins.append(model_armor)
    return plugins


# ---------------------------------------------------------------------------
# 3. Self-evaluation of the verdict
# ---------------------------------------------------------------------------

REQUIRED_SECTIONS = [
    "Product match", "Retail price (new)", "Typical used price",
    "Listing vs. market", "Red flags", "Verdict", "Confidence",
]
ALLOWED_VERDICTS = {"good deal", "fair", "overpriced", "suspicious"}


def extract_field(markdown: str, label: str) -> str | None:
    """Value of a `- **Label:** value` line in the validator markdown, or None."""
    match = re.search(rf"\*\*{re.escape(label)}:\*\*\s*(.+)", markdown or "")
    return match.group(1).strip() if match else None


def confidence_level(markdown: str) -> str | None:
    """"High" / "Medium" / "Low" from the Confidence line (without the reason), or None."""
    match = re.match(r"(high|medium|low)\b", (extract_field(markdown, "Confidence") or "").lower())
    return match.group(1).capitalize() if match else None


def evaluate_verdict(validation: str, used_search: bool) -> list[str]:
    """Rule-based checks of the validator output. Returns a list of issues."""
    issues = [f"missing section '{s}'" for s in REQUIRED_SECTIONS if extract_field(validation, s) is None]

    verdict = (extract_field(validation, "Verdict") or "").lower()
    if verdict and not any(verdict.startswith(v) for v in ALLOWED_VERDICTS):
        issues.append(f"unexpected verdict '{verdict}'")

    match = (extract_field(validation, "Product match") or "").lower()
    if match.startswith("no") and not verdict.startswith("suspicious"):
        issues.append("product does not match, but verdict is not 'Suspicious'")

    confidence = (extract_field(validation, "Confidence") or "").lower()
    if not used_search:
        issues.append("no Google Search sources were used; prices are not grounded")
        if confidence.startswith("high"):
            issues.append("confidence is 'High' without any search sources")
    return issues


def check_verdict(callback_context: CallbackContext) -> types.Content | None:
    """After-agent callback on the validator: self-evaluates the verdict.

    Stores the result in state (`verdict_check`) and, if issues are found,
    appends a visible warning to the answer.
    """
    validation = callback_context.state.get("validation")
    if not validation:
        return None  # validation skipped (e.g. not a listing)
    used_search = any(
        e.grounding_metadata and (e.grounding_metadata.grounding_chunks or e.grounding_metadata.web_search_queries)
        for e in callback_context.session.events
        if e.invocation_id == callback_context.invocation_id and e.author == "validator"
    )
    issues = evaluate_verdict(validation, used_search)
    callback_context.state["verdict_check"] = {"passed": not issues, "issues": issues}
    log_event(
        "guardrail.verdict_check", callback_context,
        level=logging.WARNING if issues else logging.INFO,
        passed=not issues, issues=issues, used_search=used_search,
        verdict=extract_field(validation, "Verdict"), confidence=confidence_level(validation),
    )
    if not issues:
        return None
    return _model_message(
        "⚠️ **Automatic quality check:** this verdict may be unreliable — "
        + "; ".join(issues) + ". Treat it with caution."
    )
