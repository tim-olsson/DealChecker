import logging
from datetime import datetime, timezone
from typing import AsyncGenerator, Literal

from google.adk.agents import BaseAgent, SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import Agent
from google.adk.apps import App
from google.adk.apps.app import EventsCompactionConfig
from google.adk.events import Event
from google.adk.tools import google_search
from google.adk.tools.google_search_tool import GoogleSearchTool
from google.adk.tools.preload_memory_tool import PreloadMemoryTool
from google.adk.utils.context_utils import Aclosing
from google.genai import types
from pydantic import BaseModel, Field

from .guardrails import (
    OUTPUT_BLOCKED_MESSAGE, SAFETY_CONFIG, build_plugins, check_verdict, confidence_level, extract_field,
)
from .observability import log_event, setup_logging, setup_tracing
from .prompts import EXTRACTOR_INSTRUCTION, FOLLOWUP_INSTRUCTION, VALIDATOR_INSTRUCTION
from .tools import DEAL_HISTORY_KEY, MAX_DEAL_HISTORY, get_deal_history

setup_logging()  # structured JSON logs for both the CLI and `adk web`
setup_tracing()  # Cloud Trace export (TRACE_TO_CLOUD); content-free spans by default

logger = logging.getLogger(__name__)

CHECK_MODEL = "gemini-3.8-flash"
FOLLOWUP_MODEL = "gemini-3.5-flash-lite"

MODEL_CONFIG = SAFETY_CONFIG.model_copy(update={
    "http_options": types.HttpOptions(retry_options=types.HttpRetryOptions(
        attempts=4, initial_delay=2.0, max_delay=30.0, http_status_codes=[408, 429, 500, 502, 503, 504],
    )),
})

NOT_A_LISTING = "not a listing"
PENDING_CLARIFICATION_KEY = "pending_clarification"  # session-scoped: one listing conversation
MAX_CLARIFICATION_ROUNDS = 2

Condition = Literal["New", "Like new", "Good", "Fair", "Poor", "For parts"]


class DealInfo(BaseModel):
    title: str = Field(description="Title of the listing")
    description: str | None = Field(description="Description of the listing")
    product_name: str = Field(description="Name of the product. Inferred from the image and title/description")
    quality: Condition | None = Field(description="Condition of the product on the scale: New, Like new, Good, Fair, Poor, For parts.")
    issues: str | None = Field(description="Issues with the product.")
    price: float | None = Field(description="Current/sale price")
    currency: str | None = Field(description="Currency code, e.g. USD, GBP, SEK")
    clarification_question: str | None = Field(description=(
        "Only if information needed to price the listing (exact product variant or asking price) "
        "cannot be determined from the images or the user's messages: one short, specific question "
        "to the user asking for it. Otherwise null."
    ))


def _deal_info(callback_context: CallbackContext) -> dict:
    """Returns the extractor output from state as a dict (empty if missing)."""
    value = callback_context.state.get("deal_info")
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, str):
        try:
            return DealInfo.model_validate_json(value).model_dump()
        except ValueError:
            return {}
    return value or {}


def _model_message(text: str) -> types.Content:
    return types.Content(role="model", parts=[types.Part(text=text)])


def _clarification_question(deal: dict, rounds: int) -> str | None:
    """Returns the question to ask the user, or None if validation can proceed."""
    if question := (deal.get("clarification_question") or "").strip():
        return question
    if deal.get("price") is None and rounds == 0:
        return "I couldn't find the asking price in the screenshot. What is the seller asking, and in which currency?"
    return None


def gate_validation(callback_context: CallbackContext) -> types.Content | None:
    """Before-agent callback on the validator: decides whether validation (and its searches) runs.

    - Not a listing: skip validation.
    - Critical info missing (human-in-the-loop clarification): stop here, ask the
      user for exactly what's missing and record `pending_clarification` in state.
      The router sends the user's next message back through the check pipeline,
      where the extractor merges the answer (or a clearer screenshot) and this
      gate runs again. At most MAX_CLARIFICATION_ROUNDS questions per listing;
      after that the check proceeds with what is known.
    """
    state = callback_context.state
    deal = _deal_info(callback_context)
    product_name = (deal.get("product_name") or "").strip().lower()
    pending = state.get(PENDING_CLARIFICATION_KEY) or {}

    if product_name == NOT_A_LISTING:
        state["validation"] = None
        if pending:
            state[PENDING_CLARIFICATION_KEY] = None
        log_event("guardrail.not_a_listing", callback_context)
        return _model_message(
            "This image doesn't look like a product listing, so I didn't validate it. "
            "Please send a screenshot or photo of a marketplace listing."
        )

    rounds = pending.get("rounds", 0)
    question = _clarification_question(deal, rounds)
    if question and rounds < MAX_CLARIFICATION_ROUNDS:
        state["validation"] = None
        state[PENDING_CLARIFICATION_KEY] = {
            "question": question,
            "rounds": rounds + 1,
            "asked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        log_event(
            "hitl.clarification_requested", callback_context,
            round=rounds + 1,
            missing_price=deal.get("price") is None,
            asked_by="extractor" if deal.get("clarification_question") else "price_backstop",
        )
        return _model_message(
            f"Before I check this listing: {question}\n\n"
            "_Reply with the missing details or send a clearer screenshot and I'll continue. "
            "If you don't know, just say so and I'll check it with what I have._"
        )

    if pending:
        log_event(
            "hitl.clarification_resolved", callback_context,
            rounds=rounds, missing_price=deal.get("price") is None,
            hit_round_limit=bool(question),
        )
        state[PENDING_CLARIFICATION_KEY] = None
    return None


def record_checked_deal(callback_context: CallbackContext) -> None:
    """Appends a compact summary of this check to the user's deal history."""
    deal = _deal_info(callback_context)
    validation = callback_context.state.get("validation") or ""
    if not deal or not validation:
        return None
    record = {
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "session_id": callback_context.session.id,
        "product_name": deal.get("product_name"),
        "title": deal.get("title"),
        "price": deal.get("price"),
        "currency": deal.get("currency"),
        "quality": deal.get("quality"),
        "verdict": extract_field(validation, "Verdict"),
        "confidence": confidence_level(validation),
        "quality_check_passed": (callback_context.state.get("verdict_check") or {}).get("passed"),
    }
    history = list(callback_context.state.get(DEAL_HISTORY_KEY) or [])
    # Assign a new list (not in-place append) so ADK records the state change.
    callback_context.state[DEAL_HISTORY_KEY] = (history + [record])[-MAX_DEAL_HISTORY:]
    return None


def _validator_answer(callback_context: CallbackContext) -> str:
    """The validator's final text answer in the current invocation."""
    for event in reversed(callback_context.session.events):
        if event.invocation_id != callback_context.invocation_id or event.author != validator_agent.name:
            continue
        if event.partial or not event.content or not event.content.parts:
            continue
        text = "".join(p.text for p in event.content.parts if p.text and not p.thought).strip()
        if text:
            return text
    return ""


def finalize_validation(callback_context: CallbackContext) -> types.Content | None:
    """After-agent callback on the validator: saves, self-checks and records the verdict.

    The validator deliberately has no `output_key`: ADK also writes content
    returned by a before_agent_callback to `output_key`, so gate messages
    ("not a listing", clarification questions) would be stored as if they were
    a validation. Saving here, after a real model answer, keeps `validation`
    empty when validation was skipped.

    One callback (instead of a list) because ADK stops a callback list at the
    first one that returns content: the self-check warning would otherwise
    prevent the deal from being recorded.
    """
    answer = _validator_answer(callback_context)
    if not answer or answer == OUTPUT_BLOCKED_MESSAGE:
        callback_context.state["validation"] = None  # nothing trustworthy to store
        return None
    callback_context.state["validation"] = answer
    warning = check_verdict(callback_context)  # writes state["verdict_check"]
    record_checked_deal(callback_context)
    return warning


extractor_agent = Agent(
    model=CHECK_MODEL,
    name='extractor_agent',
    description='Extracts structural product information from listing images.',
    instruction=EXTRACTOR_INSTRUCTION,
    output_schema=DealInfo,
    output_key="deal_info",
    generate_content_config=MODEL_CONFIG,
)

validator_agent = Agent(
    model=CHECK_MODEL,
    name="validator",
    description="Validates an extracted listing against the original product using Google Search.",
    instruction=VALIDATOR_INSTRUCTION,
    tools=[google_search],
    # No output_key: `validation` is written by finalize_validation (see its docstring).
    generate_content_config=MODEL_CONFIG,
    before_agent_callback=gate_validation,
    after_agent_callback=finalize_validation,
)

deal_check_pipeline = SequentialAgent(
    name="deal_check_pipeline",
    description="Extracts deal info from listing images and validates it via Google Search.",
    sub_agents=[extractor_agent, validator_agent],
)

followup_agent = Agent(
    model=FOLLOWUP_MODEL,
    name="followup_agent",
    description="Answers follow-up questions about checked listings using the stored results.",
    instruction=FOLLOWUP_INSTRUCTION,
    tools=[
        GoogleSearchTool(bypass_multi_tools_limit=True),
        get_deal_history,
        PreloadMemoryTool(),
    ],
    generate_content_config=MODEL_CONFIG,
)


def _has_image(content: types.Content | None) -> bool:
    return bool(content and content.parts and any(
        part.inline_data and (part.inline_data.mime_type or "").startswith("image/")
        for part in content.parts
    ))


class DealCheckerRouter(BaseAgent):
    """Routes each user message: images -> check pipeline, text only -> follow-up agent.

    Exception: while a clarification question is pending, a text reply is the
    answer to it, so it goes back through the check pipeline to resume the check.
    Deterministic (no LLM call) so routing is cheap, predictable and testable.
    """

    pipeline: BaseAgent
    followup: BaseAgent

    def __init__(self, *, name: str, pipeline: BaseAgent, followup: BaseAgent, **kwargs):
        super().__init__(
            name=name,
            pipeline=pipeline,
            followup=followup,
            sub_agents=[pipeline, followup],
            **kwargs,
        )

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        has_image = _has_image(ctx.user_content)
        pending = bool(ctx.session.state.get(PENDING_CLARIFICATION_KEY))
        target = self.pipeline if (has_image or pending) else self.followup
        log_event(
            "route.selected", ctx, agent=self.name, target=target.name,
            reason="image" if has_image else ("clarification_reply" if pending else "text_followup"),
        )
        async with Aclosing(target.run_async(ctx)) as agen:
            async for event in agen:
                yield event


async def save_session_to_memory(callback_context: CallbackContext) -> None:
    """After-agent callback: sends the session to the memory service.

    With Memory Bank, ADK ingests the events in a fire-and-forget background
    task, and memory extraction/consolidation runs asynchronously on the server,
    so the user's response is never delayed. (Short-lived processes such as the
    CLI must let that task finish before exiting; see `src/main.py`.)
    """
    try:
        await callback_context.add_session_to_memory()
    except ValueError:
        pass  # No memory service configured (e.g. local-only run).
    except Exception as exc:  # Memory is best-effort; never fail the user's turn.
        log_event("memory.submit_failed", callback_context, level=logging.WARNING, error=str(exc)[:500])
    else:
        log_event("memory.submitted", callback_context)
    return None


root_agent = DealCheckerRouter(
    name="deal_checker",
    description="Checks marketplace listings from images and answers follow-up questions.",
    pipeline=deal_check_pipeline,
    followup=followup_agent,
    after_agent_callback=save_session_to_memory,
)

app = App(
    name="deal_checker",
    root_agent=root_agent,
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=5,
        overlap_size=1,
    ),
    # Observability, input policy and Model Armor (if MODEL_ARMOR_* env vars are set).
    plugins=build_plugins(),
)
