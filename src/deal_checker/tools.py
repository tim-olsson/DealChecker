"""Function tools for DealChecker agents.

Tools return a dict with a `status` field instead of raising, so the model can
recover on its own:
  * "ok"     - `deals` holds the results.
  * "empty"  - nothing matched; `hint` says what to do next (broaden the
               filter, or ask the user for a listing).
  * "error"  - the arguments were invalid; `error` says what was wrong and
               `hint` says how to call the tool correctly.

Calls and outcomes are logged by ObservabilityPlugin (tool.intent / tool.outcome),
so the tools themselves stay pure functions of their arguments and state.
"""

from __future__ import annotations

from google.adk.tools.tool_context import ToolContext

DEAL_HISTORY_KEY = "user:checked_deals"  # `user:` prefix = shared across the user's sessions
VERDICTS = ("Good deal", "Fair", "Overpriced", "Suspicious")
MAX_DEAL_HISTORY = 50  # records kept per user (oldest dropped first)
MAX_LIMIT = MAX_DEAL_HISTORY


def _matches(record: dict, words: list[str]) -> bool:
    haystack = f"{record.get('product_name') or ''} {record.get('title') or ''}".lower()
    return all(word in haystack for word in words)


def _error(error: str, hint: str, **extra) -> dict:
    return {"status": "error", "error": error, "hint": hint, **extra}


def get_deal_history(
    tool_context: ToolContext,
    product_query: str | None = None,
    verdict: str | None = None,
    limit: int = 10,
) -> dict:
    """Looks up listings this user checked before (across all their sessions).

    Use this for questions about earlier checks, for example "which of my deals
    is best?", "have I looked at a Barista Express before?", "what did I check
    last week?" or "compare this one with the others". Do NOT use it for the
    current listing: its extracted data and validation are already in your
    instructions. Do NOT use Google Search for this; the history only exists here.

    Args:
        product_query: Optional words to filter by product, matched
            case-insensitively against the product name and listing title. Every
            word must match (e.g. "barista express"). Leave empty for all
            products.
        verdict: Optional verdict filter. One of "Good deal", "Fair",
            "Overpriced", "Suspicious". Leave empty for all verdicts.
        limit: Maximum number of deals to return, most recent first. 1 to 50,
            default 10.

    Returns:
        A dict with:
          status: "ok", "empty" or "error".
          deals: (status "ok") list of past checks, most recent first, each with
            checked_at (ISO 8601, UTC), product_name, title, price, currency,
            quality (condition), verdict, confidence, quality_check_passed
            (False means the automatic self-check flagged that verdict as
            unreliable) and is_current_listing (True for the listing in this
            conversation).
          total_matches: (status "ok") how many deals matched before `limit`.
          total_checked: how many deals the user has checked in total.
          error: (status "error") what was wrong with the arguments.
          hint: (status "empty"/"error") what to do next.
    """
    if verdict:
        canonical = next((v for v in VERDICTS if v.lower() == verdict.strip().lower()), None)
        if canonical is None:
            return _error(
                f"Unknown verdict {verdict!r}.",
                f"Use one of {', '.join(VERDICTS)}, or leave verdict empty to include all verdicts.",
                allowed_verdicts=list(VERDICTS),
            )
        verdict = canonical
    if not isinstance(limit, int) or not 1 <= limit <= MAX_LIMIT:
        return _error(
            f"limit must be a whole number between 1 and {MAX_LIMIT} (got {limit!r}).",
            f"Retry with a limit between 1 and {MAX_LIMIT}, e.g. 10.",
        )

    history = list(tool_context.state.get(DEAL_HISTORY_KEY) or [])
    if not history:
        return {
            "status": "empty",
            "total_checked": 0,
            "hint": "The user has not checked any listings yet. Tell them, and invite them to send a "
                    "screenshot of a listing to check. Do not invent past deals.",
        }

    words = (product_query or "").lower().split()
    matches = [
        r for r in reversed(history)  # most recent first
        if _matches(r, words) and (not verdict or (r.get("verdict") or "").lower().startswith(verdict.lower()))
    ]
    if not matches:
        filters = " and ".join(
            f for f in (f"product {product_query!r}" if words else "", f"verdict {verdict!r}" if verdict else "") if f
        )
        return {
            "status": "empty",
            "total_checked": len(history),
            "hint": f"None of the user's {len(history)} checked deals match {filters}. Retry with fewer or "
                    "shorter product words (e.g. the brand only) or without the verdict filter, or tell the "
                    "user nothing matched.",
        }

    session_id = tool_context.session.id
    deals = [
        {
            "checked_at": r.get("checked_at"),
            "product_name": r.get("product_name"),
            "title": r.get("title"),
            "price": r.get("price"),
            "currency": r.get("currency"),
            "quality": r.get("quality"),
            "verdict": r.get("verdict"),
            "confidence": r.get("confidence"),
            "quality_check_passed": r.get("quality_check_passed"),
            "is_current_listing": r.get("session_id") == session_id,
        }
        for r in matches[:limit]
    ]
    return {"status": "ok", "deals": deals, "total_matches": len(matches), "total_checked": len(history)}
