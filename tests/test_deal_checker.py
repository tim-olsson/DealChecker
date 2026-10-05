"""Offline unit tests for deterministic logic: guardrails, clarification gate,
validation finalization and app wiring. No model calls; run with `uv run pytest`."""

from types import SimpleNamespace

import pytest
from google.adk.events import Event
from google.genai import types

from deal_checker import agent as dc
from deal_checker import guardrails as g

GOOD_VALIDATION = """- **Product match:** Yes - same model
- **Observed in listing:** black machine
- **Retail price (new):** £479 (John Lewis)
- **Typical used price:** £250-£350 (eBay sold)
- **Listing vs. market:** 7% above midpoint
- **Red flags:** None found
- **Verdict:** Fair
- **Confidence:** High - clear match"""

DEAL = {"title": "Sage Barista Express", "description": None, "product_name": "Sage Barista Express",
        "quality": "Like new", "issues": None, "price": 320.0, "currency": "GBP", "clarification_question": None}


def _ctx(state: dict, events: list | None = None, invocation_id: str = "inv-1"):
    """Minimal stand-in for ADK's CallbackContext (state, session.events, ids)."""
    return SimpleNamespace(
        state=state, invocation_id=invocation_id, agent_name="validator",
        session=SimpleNamespace(id="s-1", events=events or []),
    )


def _validator_event(text: str, *, searched: bool = True, invocation_id: str = "inv-1") -> Event:
    return Event(
        invocation_id=invocation_id, author="validator",
        content=types.Content(role="model", parts=[types.Part(text=text)]),
        grounding_metadata=types.GroundingMetadata(web_search_queries=["sage barista express used price"])
        if searched else None,
    )


# --- Verdict self-evaluation -------------------------------------------------

class TestEvaluateVerdict:
    def test_complete_grounded_verdict_passes(self):
        assert g.evaluate_verdict(GOOD_VALIDATION, used_search=True) == []

    def test_mismatch_must_be_suspicious(self):
        issues = g.evaluate_verdict(GOOD_VALIDATION.replace("Yes - same", "No - different"), True)
        assert any("Suspicious" in i for i in issues)

    def test_ungrounded_high_confidence_is_flagged(self):
        issues = g.evaluate_verdict(GOOD_VALIDATION, used_search=False)
        assert len(issues) == 2  # no sources + High confidence without sources

    def test_unknown_verdict_is_flagged(self):
        issues = g.evaluate_verdict(GOOD_VALIDATION.replace("Fair", "Amazing"), True)
        assert any("unexpected verdict" in i for i in issues)

    def test_missing_sections_are_listed(self):
        issues = g.evaluate_verdict("**Verdict:** Fair", True)
        assert len([i for i in issues if i.startswith("missing")]) == 6


# --- Input policy --------------------------------------------------------------

def _image(mime: str = "image/png", size: int = 10) -> types.Part:
    return types.Part(inline_data=types.Blob(mime_type=mime, data=b"x" * size))


@pytest.mark.parametrize(
    "parts, expected",
    [
        ([_image()], None),
        ([_image()] * 5, None),
        ([_image()] * 6, "at most"),
        ([_image("application/pdf")], "Unsupported"),
        ([_image(size=8 * 1024 * 1024)], "larger"),
        ([types.Part(text="just text")], None),
    ],
)
def test_input_policy(parts, expected):
    reason = g.check_input_policy(types.Content(role="user", parts=parts))
    assert (reason is None) if expected is None else (expected in reason)


# --- Clarification gate (human-in-the-loop) -------------------------------------

def _gate(deal: dict, pending: dict | None = None):
    state = {"deal_info": deal, dc.PENDING_CLARIFICATION_KEY: pending}
    message = dc.gate_validation(_ctx(state))
    return message, state


class TestGateValidation:
    def test_complete_listing_proceeds(self):
        message, state = _gate(DEAL)
        assert message is None and not state[dc.PENDING_CLARIFICATION_KEY]

    def test_missing_price_pauses_and_asks(self):
        message, state = _gate({**DEAL, "price": None})
        assert message is not None and "price" in message.parts[0].text.lower()
        assert state[dc.PENDING_CLARIFICATION_KEY]["rounds"] == 1
        assert state["validation"] is None

    def test_user_does_not_know_price_proceeds(self):
        message, state = _gate({**DEAL, "price": None}, pending={"rounds": 1})
        assert message is None and state[dc.PENDING_CLARIFICATION_KEY] is None

    def test_extractor_question_is_asked(self):
        message, _ = _gate({**DEAL, "clarification_question": "Which model is it?"}, pending={"rounds": 1})
        assert "Which model is it?" in message.parts[0].text

    def test_round_limit_proceeds(self):
        message, state = _gate({**DEAL, "clarification_question": "Which model?"}, pending={"rounds": 2})
        assert message is None and state[dc.PENDING_CLARIFICATION_KEY] is None

    def test_not_a_listing_skips_and_clears(self):
        message, state = _gate({**DEAL, "product_name": "not a listing"}, pending={"rounds": 1})
        assert "doesn't look like a product listing" in message.parts[0].text
        assert state[dc.PENDING_CLARIFICATION_KEY] is None and state["validation"] is None


# --- Validation finalization (regression tests) -----------------------------------

class TestFinalizeValidation:
    def test_saves_checks_and_records(self):
        state = {"deal_info": DEAL}
        warning = dc.finalize_validation(_ctx(state, [_validator_event(GOOD_VALIDATION)]))
        assert warning is None
        assert state["validation"] == GOOD_VALIDATION
        assert state["verdict_check"] == {"passed": True, "issues": []}
        record = state[dc.DEAL_HISTORY_KEY][-1]
        assert record["verdict"] == "Fair" and record["quality_check_passed"] is True

    def test_failed_self_check_still_records_deal(self):
        # Regression: with a callback list, the warning stopped record_checked_deal.
        state = {"deal_info": DEAL}
        warning = dc.finalize_validation(_ctx(state, [_validator_event(GOOD_VALIDATION, searched=False)]))
        assert warning is not None and "quality check" in warning.parts[0].text
        assert state[dc.DEAL_HISTORY_KEY][-1]["quality_check_passed"] is False

    def test_model_armor_block_is_not_stored(self):
        state = {"deal_info": DEAL}
        dc.finalize_validation(_ctx(state, [_validator_event(g.OUTPUT_BLOCKED_MESSAGE)]))
        assert state["validation"] is None and dc.DEAL_HISTORY_KEY not in state

    def test_ignores_events_from_other_invocations(self):
        state = {"deal_info": DEAL}
        old = _validator_event(GOOD_VALIDATION, invocation_id="older")
        dc.finalize_validation(_ctx(state, [old]))
        assert state["validation"] is None

    def test_validator_has_no_output_key(self):
        assert dc.validator_agent.output_key is None


# --- App wiring --------------------------------------------------------------------

def test_app_wiring():
    assert dc.app.name == "deal_checker"
    plugin_names = [p.name for p in dc.app.plugins]
    assert plugin_names[:2] == ["deal_checker_observability", "deal_checker_policy"]
    assert [a.name for a in dc.deal_check_pipeline.sub_agents] == ["extractor_agent", "validator"]
    for llm_agent in (dc.extractor_agent, dc.validator_agent, dc.followup_agent):
        assert llm_agent.generate_content_config.safety_settings, llm_agent.name
