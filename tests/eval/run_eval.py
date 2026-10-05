"""DealChecker evaluation harness: runs the golden dataset against the real app.

    uv run python tests/eval/run_eval.py                       # all cases, once
    uv run python tests/eval/run_eval.py --repeat 3            # pass rate over 3 runs
    uv run python tests/eval/run_eval.py --case fair_listing --no-judge
    uv run python tests/eval/run_eval.py --baseline tests/eval/baseline.json

Each case in golden.json is a conversation. The app runs in-process with the
real models, Google Search and Model Armor, but with in-memory sessions and no
Memory Bank, so eval runs never touch your stored sessions or memories.

Grading per turn:
  * deterministic checks (route, validation ran, extracted price, verdict range,
    clarification state, guardrail blocks, verdict self-check) - graded by code;
  * optional `rubric` - graded by an LLM judge (Flash-Lite, temperature 0).
Exit code is 1 if any case fails (pass rate < 100%), so it can gate CI.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
# Keep eval runs out of the production telemetry unless explicitly requested.
if "--telemetry" not in sys.argv:
    os.environ["LOG_TO_CLOUD"] = "false"
    os.environ["TRACE_TO_CLOUD"] = "false"
    os.environ["LOG_LEVEL"] = "WARNING"
    os.environ["LOG_FORMAT"] = "text"

from google import genai  # noqa: E402
from google.adk.runners import Runner  # noqa: E402
from google.adk.sessions import InMemorySessionService  # noqa: E402
from google.genai import types  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from deal_checker.agent import PENDING_CLARIFICATION_KEY, DealInfo, app  # noqa: E402
from deal_checker.guardrails import (  # noqa: E402
    INPUT_BLOCKED_MESSAGE,
    OUTPUT_BLOCKED_MESSAGE,
    extract_field,
    check_input_policy,
)

DATASET = EVAL_DIR / "golden.json"
FIXTURES = EVAL_DIR / "fixtures"
RESULTS = EVAL_DIR / "results"
JUDGE_MODEL = "gemini-3.5-flash-lite"
USER_ID = "eval"


# ---------------------------------------------------------------------------
# Running a conversation
# ---------------------------------------------------------------------------


def build_message(turn: dict) -> types.Content:
    parts = [types.Part(text=turn.get("text") or "Check this listing.")]
    for name in turn.get("images", []):
        parts.append(types.Part.from_bytes(data=(FIXTURES / name).read_bytes(), mime_type="image/png"))
    return types.Content(role="user", parts=parts)


def _deal_info(state: dict) -> dict:
    value = state.get("deal_info")
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, str):
        try:
            return DealInfo.model_validate_json(value).model_dump()
        except ValueError:
            return {}
    return value or {}


async def run_turn(runner: Runner, session_id: str, turn: dict) -> dict:
    """Runs one user turn and returns what the graders need."""
    message = build_message(turn)
    authors: list[str] = []
    final_texts: list[str] = []
    tools_called: list[str] = []
    started = time.monotonic()
    async for event in runner.run_async(user_id=USER_ID, session_id=session_id, new_message=message):
        authors.append(event.author)
        tools_called.extend(fc.name for fc in event.get_function_calls())
        # The extractor's output is internal structured data (graded via state), not user-facing prose.
        if event.author == "extractor_agent":
            continue
        if event.is_final_response() and event.content and event.content.parts:
            text = "".join(p.text or "" for p in event.content.parts if not p.thought)
            if text:
                final_texts.append(text)
    session = await runner.session_service.get_session(app_name=app.name, user_id=USER_ID, session_id=session_id)
    state = dict(session.state) if session else {}
    if "extractor_agent" in authors:
        route = "deal_check_pipeline"
    elif "followup_agent" in authors:
        route = "followup_agent"
    else:
        route = "none"
    return {
        "message": message,
        "route": route,
        "response": final_texts[-1] if final_texts else "",
        "all_text": "\n".join(final_texts),
        "state": state,
        "tools_called": tools_called,
        "duration_s": round(time.monotonic() - started, 1),
    }


# ---------------------------------------------------------------------------
# Deterministic checks
# ---------------------------------------------------------------------------


def check_expectations(expect: dict, out: dict) -> list[dict]:
    """Returns one {check, passed, detail} per expectation."""
    state, deal = out["state"], _deal_info(out["state"])
    validation = state.get("validation") or ""
    verdict = (extract_field(validation, "Verdict") or "").strip()
    results = []

    def add(name: str, passed: bool, detail: Any) -> None:
        results.append({"check": name, "passed": bool(passed), "detail": detail})

    for key, want in expect.items():
        if key == "route":
            add(key, out["route"] == want, f"got {out['route']}")
        elif key == "validated":
            add(key, bool(validation) == want, f"validation {'present' if validation else 'absent'}")
        elif key == "product_contains":
            got = (deal.get("product_name") or "")
            add(key, want.lower() in got.lower(), f"product_name={got!r}")
        elif key == "price":
            got = deal.get("price")
            add(key, got is not None and abs(float(got) - want) < 0.01, f"price={got}")
        elif key == "currency":
            add(key, (deal.get("currency") or "").upper() == want, f"currency={deal.get('currency')}")
        elif key == "verdict_in":
            add(key, any(verdict.lower().startswith(v.lower()) for v in want), f"verdict={verdict!r}")
        elif key == "verdict_check_passed":
            vc = state.get("verdict_check") or {}
            add(key, vc.get("passed") == want, f"verdict_check={vc}")
        elif key == "clarification_pending":
            pending = state.get(PENDING_CLARIFICATION_KEY)
            add(key, bool(pending) == want, f"pending={pending}")
        elif key == "response_contains_any":
            text = out["all_text"].lower()
            add(key, any(s.lower() in text for s in want), f"response={out['response'][:120]!r}")
        elif key == "blocked_by":
            if want == "model_armor":
                passed = out["response"].strip() in {INPUT_BLOCKED_MESSAGE, OUTPUT_BLOCKED_MESSAGE}
            elif want == "input_policy":
                reason = check_input_policy(out["message"])
                passed = bool(reason) and out["response"].strip() == reason
            else:
                passed = False
            add(key, passed, f"response={out['response'][:120]!r}")
        elif key == "tools_called":
            add(key, set(want) <= set(out["tools_called"]), f"called={out['tools_called']}")
        else:
            add(key, False, "unknown expectation key")
    return results


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------


class JudgeVerdict(BaseModel):
    passed: bool
    reason: str


JUDGE_PROMPT = """You are grading one reply of DealChecker, an assistant that checks
second-hand marketplace listings. Grade ONLY against the rubric. Be strict but fair:
pass only if the reply clearly satisfies every part of the rubric.

User message:
{user}

Extracted listing data (from the screenshot, may be empty):
{deal}

Assistant reply:
{reply}

Rubric:
{rubric}
"""


async def judge(client: genai.Client, rubric: str, out: dict) -> dict:
    user = "".join(p.text or "" for p in out["message"].parts if p.text)
    images = sum(1 for p in out["message"].parts if p.inline_data)
    prompt = JUDGE_PROMPT.format(
        user=f"{user} (+{images} image(s))" if images else user,
        deal=json.dumps(_deal_info(out["state"]), ensure_ascii=False),
        reply=out["all_text"],
        rubric=rubric,
    )
    try:
        resp = await client.aio.models.generate_content(
            model=JUDGE_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0, response_mime_type="application/json", response_schema=JudgeVerdict
            ),
        )
        verdict = JudgeVerdict.model_validate_json(resp.text)
        return {"check": "rubric", "passed": verdict.passed, "detail": verdict.reason}
    except Exception as exc:  # A judge failure is a failed check, not a crash.
        return {"check": "rubric", "passed": False, "detail": f"judge error: {exc}"}


# ---------------------------------------------------------------------------
# Orchestration and reporting
# ---------------------------------------------------------------------------


async def run_case(case: dict, client: genai.Client | None) -> dict:
    runner = Runner(app=app, session_service=InMemorySessionService())
    session = await runner.session_service.create_session(app_name=app.name, user_id=USER_ID)
    turns = []
    for index, turn in enumerate(case["turns"], start=1):
        try:
            out = await run_turn(runner, session.id, turn)
        except Exception as exc:
            turns.append({"turn": index, "passed": False, "checks": [
                {"check": "run", "passed": False, "detail": f"{type(exc).__name__}: {exc}"}]})
            break
        if "expect_any" in turn:
            alternatives = [check_expectations(e, out) for e in turn["expect_any"]]
            best = max(alternatives, key=lambda cs: (all(c["passed"] for c in cs), sum(c["passed"] for c in cs)))
            checks = [{**c, "check": f"any:{c['check']}"} for c in best]
        else:
            checks = check_expectations(turn.get("expect", {}), out)
        if client and turn.get("rubric"):
            checks.append(await judge(client, turn["rubric"], out))
        turns.append({
            "turn": index,
            "passed": all(c["passed"] for c in checks),
            "route": out["route"],
            "duration_s": out["duration_s"],
            "response_preview": out["response"][:300],
            "checks": checks,
        })
    return {"id": case["id"], "passed": all(t["passed"] for t in turns), "turns": turns}


async def run_all(cases: list[dict], repeat: int, concurrency: int, use_judge: bool) -> list[dict]:
    client = genai.Client() if use_judge else None
    semaphore = asyncio.Semaphore(concurrency)

    async def guarded(case: dict, run: int) -> dict:
        async with semaphore:
            result = await run_case(case, client)
            mark = "PASS" if result["passed"] else "FAIL"
            print(f"  [{mark}] {case['id']} (run {run})", flush=True)
            return {**result, "run": run}

    return await asyncio.gather(*(guarded(c, r) for r in range(1, repeat + 1) for c in cases))


def summarize(cases: list[dict], runs: list[dict]) -> dict[str, float]:
    print("\n=== DealChecker eval results ===")
    rates: dict[str, float] = {}
    for case in cases:
        case_runs = [r for r in runs if r["id"] == case["id"]]
        rate = sum(r["passed"] for r in case_runs) / len(case_runs)
        rates[case["id"]] = rate
        print(f"{'PASS' if rate == 1 else 'FAIL':4}  {rate:5.0%}  {case['id']}")
        for run in case_runs:
            for turn in run["turns"]:
                for check in turn["checks"]:
                    if not check["passed"]:
                        print(f"        run {run['run']} turn {turn['turn']}: {check['check']} -> {check['detail']}")
    passed = sum(rate == 1 for rate in rates.values())
    print(f"\n{passed}/{len(rates)} cases passed; overall check pass rate "
          f"{sum(c['passed'] for r in runs for t in r['turns'] for c in t['checks']) / max(1, sum(len(t['checks']) for r in runs for t in r['turns'])):.0%}")
    return rates


def compare(rates: dict[str, float], baseline_path: Path) -> None:
    baseline = json.loads(baseline_path.read_text())["pass_rates"]
    print(f"\n=== Compared with {baseline_path.name} ===")
    for case_id, rate in rates.items():
        before = baseline.get(case_id)
        if before is None:
            print(f"  new        {case_id}: {rate:.0%}")
        elif rate < before:
            print(f"  REGRESSED  {case_id}: {before:.0%} -> {rate:.0%}")
        elif rate > before:
            print(f"  improved   {case_id}: {before:.0%} -> {rate:.0%}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--case", action="append", help="Run only this case id (repeatable).")
    parser.add_argument("--repeat", type=int, default=1, help="Runs per case (measures flakiness).")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM-judge rubric checks.")
    parser.add_argument("--baseline", type=Path, help="Previous results JSON to compare against.")
    parser.add_argument("--telemetry", action="store_true", help="Keep Cloud Logging/Trace settings from .env.")
    args = parser.parse_args()

    cases = json.loads(DATASET.read_text())["cases"]
    if args.case:
        cases = [c for c in cases if c["id"] in set(args.case)]
        if not cases:
            parser.error(f"no such case: {args.case}")

    print(f"Running {len(cases)} case(s) x {args.repeat} ...")
    started = time.monotonic()
    runs = asyncio.run(run_all(cases, args.repeat, args.concurrency, not args.no_judge))
    rates = summarize(cases, runs)

    RESULTS.mkdir(exist_ok=True)
    out_path = RESULTS / f"results_{datetime.now():%Y%m%d_%H%M%S}.json"
    out_path.write_text(json.dumps({
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "duration_s": round(time.monotonic() - started, 1),
        "judge_model": None if args.no_judge else JUDGE_MODEL,
        "repeat": args.repeat,
        "pass_rates": rates,
        "runs": runs,
    }, indent=2, ensure_ascii=False))
    print(f"Results: {out_path.relative_to(ROOT)}")
    if args.baseline:
        compare(rates, args.baseline)
    return 0 if all(rate == 1 for rate in rates.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
