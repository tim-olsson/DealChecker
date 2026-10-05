"""Offline tests for the get_deal_history tool and the CLI error guidance."""

import inspect
from types import SimpleNamespace

import pytest
from google.auth import exceptions as auth_exceptions
from google.genai import errors as genai_errors

from deal_checker import agent as dc
from deal_checker import errors as e
from deal_checker.tools import DEAL_HISTORY_KEY, get_deal_history

HISTORY = [
    {"checked_at": "2026-10-01T10:00:00+00:00", "session_id": "s-old", "product_name": "Sage Barista Express",
     "title": "Barista Express black", "price": 320, "currency": "GBP", "quality": "Like new",
     "verdict": "Fair", "confidence": "High", "quality_check_passed": True},
    {"checked_at": "2026-10-01T11:00:00+00:00", "session_id": "s-old2", "product_name": "DeLonghi Dedica",
     "title": "Dedica EC685", "price": 40, "currency": "GBP", "quality": "Good",
     "verdict": "Suspicious - too cheap", "confidence": "Medium", "quality_check_passed": False},
    {"checked_at": "2026-10-02T09:00:00+00:00", "session_id": "s-1", "product_name": "Sage Barista Pro",
     "title": "Barista Pro", "price": 450, "currency": "GBP", "quality": "Good",
     "verdict": "Good deal", "confidence": "High", "quality_check_passed": True},
]


def _tool_ctx(history=None):
    state = {DEAL_HISTORY_KEY: history} if history is not None else {}
    return SimpleNamespace(state=state, invocation_id="inv-1", agent_name="followup_agent",
                           session=SimpleNamespace(id="s-1", events=[]), user_id="me")


# --- get_deal_history ----------------------------------------------------------

class TestGetDealHistory:
    def test_returns_most_recent_first_and_flags_current_listing(self):
        result = get_deal_history(_tool_ctx(HISTORY))
        assert result["status"] == "ok"
        assert [d["product_name"] for d in result["deals"]][0] == "Sage Barista Pro"
        assert result["deals"][0]["is_current_listing"] is True
        assert "session_id" not in result["deals"][0]  # internal ids are not exposed to the model

    def test_product_query_requires_all_words(self):
        result = get_deal_history(_tool_ctx(HISTORY), product_query="sage barista")
        assert {d["product_name"] for d in result["deals"]} == {"Sage Barista Express", "Sage Barista Pro"}
        assert get_deal_history(_tool_ctx(HISTORY), product_query="sage dedica")["status"] == "empty"

    def test_verdict_filter_is_case_insensitive_and_prefix_matched(self):
        result = get_deal_history(_tool_ctx(HISTORY), verdict="suspicious")
        assert [d["product_name"] for d in result["deals"]] == ["DeLonghi Dedica"]

    def test_limit(self):
        result = get_deal_history(_tool_ctx(HISTORY), limit=1)
        assert len(result["deals"]) == 1 and result["total_matches"] == 3

    @pytest.mark.parametrize("kwargs, needle", [
        ({"verdict": "bargain"}, "Use one of"),
        ({"limit": 0}, "between 1 and 50"),
        ({"limit": 500}, "between 1 and 50"),
    ])
    def test_invalid_arguments_return_guided_error(self, kwargs, needle):
        result = get_deal_history(_tool_ctx(HISTORY), **kwargs)
        assert result["status"] == "error" and result["error"] and needle in result["hint"]

    def test_no_history_tells_model_not_to_invent(self):
        result = get_deal_history(_tool_ctx())
        assert result["status"] == "empty" and "Do not invent" in result["hint"]

    def test_no_match_suggests_broadening(self):
        result = get_deal_history(_tool_ctx(HISTORY), product_query="rocket appartamento")
        assert result["status"] == "empty" and "fewer" in result["hint"] and result["total_checked"] == 3

    def test_docstring_documents_every_argument_and_return_status(self):
        doc = inspect.getdoc(get_deal_history)
        for word in ("product_query", "verdict", "limit", "Returns", '"ok"', '"empty"', '"error"', "hint"):
            assert word in doc, word

    def test_wired_into_followup_agent(self):
        assert get_deal_history in dc.followup_agent.tools
        search = next(t for t in dc.followup_agent.tools if getattr(t, "name", "") == "google_search")
        assert search.bypass_multi_tools_limit  # built-in search can't share a request with function tools


# --- CLI error guidance --------------------------------------------------------

def _api_error(code: int, message: str, status: str = "") -> genai_errors.APIError:
    cls = genai_errors.ClientError if code < 500 else genai_errors.ServerError
    return cls(code, {"error": {"code": code, "message": message, "status": status}})


@pytest.mark.parametrize("exc, expected_in_hint, retryable", [
    (_api_error(400, "API key not valid. Please pass a valid API key.", "INVALID_ARGUMENT"), "GOOGLE_API_KEY", False),
    (_api_error(403, "Model Armor API has not been used in project 123", "PERMISSION_DENIED"), "enable_apis.sh", False),
    (_api_error(403, "Caller does not have permission", "PERMISSION_DENIED"), "application-default login", False),
    (_api_error(404, "projects/1/locations/x/reasoningEngines/9 not found", "NOT_FOUND"), "create_agent_engine.py", False),
    (_api_error(429, "Resource exhausted", "RESOURCE_EXHAUSTED"), "Quotas", True),
    (_api_error(503, "The model is overloaded", "UNAVAILABLE"), "few minutes", True),
    (auth_exceptions.DefaultCredentialsError("no creds"), "application-default login", False),
    (auth_exceptions.MutualTLSChannelError("cert provider -11"), "GOOGLE_API_USE_CLIENT_CERTIFICATE=false", False),
    (ValueError("Unsupported SESSION_SERVICE_URI scheme: 'foo'"), "agentengine://", False),
    (RuntimeError("boom"), "--debug", False),
])
def test_explain_gives_actionable_guidance(exc, expected_in_hint, retryable):
    guided = e.explain(exc)
    assert expected_in_hint in guided.hint
    assert guided.retryable is retryable
    assert guided.render().startswith("Error: ")


def test_check_config_flags_missing_credentials(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    problems = e.check_config()
    assert problems and "GOOGLE_API_KEY" in problems[0].hint


def test_check_config_ok_with_key(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.delenv("SESSION_SERVICE_URI", raising=False)
    monkeypatch.delenv("MEMORY_SERVICE_URI", raising=False)
    assert e.check_config() == []


def test_transient_errors_are_retried():
    retry = dc.MODEL_CONFIG.http_options.retry_options
    assert retry.attempts >= 3 and {429, 503} <= set(retry.http_status_codes)
    for llm_agent in (dc.extractor_agent, dc.validator_agent, dc.followup_agent):
        assert llm_agent.generate_content_config is dc.MODEL_CONFIG, llm_agent.name


# --- Guardrail helpers and tool logging ------------------------------------------

@pytest.mark.parametrize("line, expected", [
    ("- **Confidence:** High – clear match and consistent prices", "High"),
    ("- **Confidence:** medium: few sources", "Medium"),
    ("- **Confidence:** Low", "Low"),
    ("- **Confidence:** unsure", None),
    ("no confidence line", None),
])
def test_confidence_level(line, expected):
    from deal_checker.guardrails import confidence_level
    assert confidence_level(line) == expected


def test_search_subagent_output_not_screened_twice():
    from deal_checker.guardrails import SKIP_RESPONSE_AGENTS
    assert "google_search_agent" in SKIP_RESPONSE_AGENTS


def test_observability_plugin_logs_tool_calls_without_argument_values(caplog):
    import asyncio
    import logging

    from deal_checker.observability import ObservabilityPlugin

    plugin = ObservabilityPlugin()
    tool = SimpleNamespace(name="get_deal_history")
    ctx = SimpleNamespace(invocation_id="inv-1", function_call_id="fc-1")
    args = {"product_query": "secret user text", "verdict": None, "limit": 5}
    logger = logging.getLogger("deal_checker")
    logger.addHandler(caplog.handler)
    old_level = logger.level
    logger.setLevel(logging.INFO)
    try:
        asyncio.run(plugin.before_tool_callback(tool=tool, tool_args=args, tool_context=ctx))
        asyncio.run(plugin.after_tool_callback(
            tool=tool, tool_args=args, tool_context=ctx, result={"status": "ok", "deals": [{}, {}]},
        ))
    finally:
        logger.removeHandler(caplog.handler)
        logger.setLevel(old_level)
    events = {r.getMessage(): r.json_fields for r in caplog.records if hasattr(r, "json_fields")}
    assert events["tool.intent"]["args"] == ["limit", "product_query"]
    assert "secret user text" not in str(events)
    assert events["tool.outcome"]["status"] == "ok" and events["tool.outcome"]["results"] == 2
    assert events["tool.outcome"]["duration_ms"] is not None
