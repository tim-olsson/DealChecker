"""Command-line entry point for DealChecker.

Examples (run from the project root):
    uv run src/main.py screenshots/sample.png                 # new check (new session)
    uv run src/main.py photo1.png photo2.jpg                  # several photos of one listing
    uv run src/main.py --resume -q "What should I offer?"     # follow-up in latest session
    uv run src/main.py -q "Which of my checked deals is best?"  # question in a new session
"""

import argparse
import asyncio
import logging
import mimetypes
import os
import sys
import traceback
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from google.adk.cli.service_registry import get_service_registry
from google.adk.memory import BaseMemoryService
from google.adk.runners import Runner
from google.adk.sessions import BaseSessionService, Session
from google.genai import types

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

from deal_checker.agent import PENDING_CLARIFICATION_KEY, app  # noqa: E402
from deal_checker.errors import check_config, explain  # noqa: E402
from deal_checker.guardrails import ALLOWED_IMAGE_TYPES, MAX_IMAGE_BYTES, MAX_IMAGES  # noqa: E402
from deal_checker.observability import log_event  # noqa: E402

USER_ID = "me"
DEFAULT_IMAGE_PROMPT = "Check this listing."

DEFAULT_SESSION_SERVICE_URI = f"sqlite:///{PROJECT_ROOT / '.adk' / 'sessions.db'}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check second-hand listings from images.")
    parser.add_argument("images", nargs="*", type=Path, help="Listing image file(s) to check.")
    parser.add_argument("-q", "--question", help="Question or instruction to send.")
    parser.add_argument(
        "--resume", action="store_true",
        help="Continue the most recent session (for follow-up questions).",
    )
    parser.add_argument("--debug", action="store_true", help="Show full tracebacks on errors.")
    args = parser.parse_args()
    if not args.images and not args.question:
        parser.error(
            "provide at least one image or a --question, e.g.\n"
            "  uv run src/main.py screenshots/sample.png\n"
            "  uv run src/main.py --resume -q \"What should I offer?\""
        )
    if len(args.images) > MAX_IMAGES:
        parser.error(f"at most {MAX_IMAGES} images per check (got {len(args.images)}); pick the clearest ones.")
    for image in args.images:
        if not image.is_file():
            hint = f" Did you mean screenshots/{image.name}?" if (PROJECT_ROOT / "screenshots" / image.name).is_file() else ""
            parser.error(f"image not found: {image} (paths are relative to {Path.cwd()}).{hint}")
        mime_type = mimetypes.guess_type(image.name)[0]
        if mime_type not in ALLOWED_IMAGE_TYPES:
            parser.error(
                f"unsupported image type for {image.name} ({mime_type or 'unknown'}); use JPEG, PNG or WebP. "
                "For iPhone HEIC photos, export/share as JPEG or take a screenshot."
            )
        if image.stat().st_size > MAX_IMAGE_BYTES:
            parser.error(
                f"{image.name} is {image.stat().st_size / 1_048_576:.1f} MB; the limit is "
                f"{MAX_IMAGE_BYTES // 1_048_576} MB. Send a screenshot or a resized copy."
            )
    return args


def create_session_service() -> BaseSessionService:
    uri = os.getenv("SESSION_SERVICE_URI", DEFAULT_SESSION_SERVICE_URI)
    parsed = urlparse(uri)
    if parsed.scheme == "sqlite" and parsed.path:
        # SQLite does not create missing directories.
        Path(parsed.path[1:]).parent.mkdir(parents=True, exist_ok=True)

    service = get_service_registry().create_session_service(
        uri, agents_dir=str(PROJECT_ROOT)
    )
    if service is None:
        raise ValueError(f"Unsupported SESSION_SERVICE_URI scheme: {parsed.scheme!r}")
    return service


def create_memory_service() -> BaseMemoryService | None:
    """Long-term memory from MEMORY_SERVICE_URI (e.g. agentengine://...); None if unset."""
    uri = os.getenv("MEMORY_SERVICE_URI")
    if not uri:
        return None
    service = get_service_registry().create_memory_service(
        uri, agents_dir=str(PROJECT_ROOT)
    )
    if service is None:
        raise ValueError(f"Unsupported MEMORY_SERVICE_URI scheme: {urlparse(uri).scheme!r}")
    return service


async def get_session(service: BaseSessionService, resume: bool) -> Session:
    """Returns the latest session if resuming (and one exists), otherwise a new one."""
    if resume:
        existing = await service.list_sessions(app_name=app.name, user_id=USER_ID)
        if existing.sessions:
            latest = max(existing.sessions, key=lambda s: s.last_update_time)
            print(f"Resuming session {latest.id}")
            return await service.get_session(
                app_name=app.name, user_id=USER_ID, session_id=latest.id
            )
        print("No previous session found; starting a new one.")
    session = await service.create_session(app_name=app.name, user_id=USER_ID)
    print(f"Created session {session.id}")
    return session


def build_message(images: list[Path], question: str | None) -> types.Content:
    text = question or DEFAULT_IMAGE_PROMPT
    parts = [types.Part(text=text)]
    for image in images:
        mime_type = mimetypes.guess_type(image.name)[0] or "image/png"
        parts.append(types.Part.from_bytes(data=image.read_bytes(), mime_type=mime_type))
    return types.Content(role="user", parts=parts)


STOP_REASON_HINTS = {
    "SAFETY": "The answer was blocked by safety filters. Try a different screenshot or rephrase the question.",
    "MAX_TOKENS": "The answer was cut off. Ask a narrower question.",
    "RECITATION": "The answer was blocked for quoting copyrighted text. Rephrase the question.",
}


async def run(args: argparse.Namespace) -> None:
    session_service = create_session_service()
    runner = Runner(
        app=app,
        session_service=session_service,
        memory_service=create_memory_service(),
    )
    session = await get_session(session_service, args.resume)

    msg = build_message(args.images, args.question)
    async for event in runner.run_async(user_id=USER_ID, session_id=session.id, new_message=msg):
        text = ""
        if event.is_final_response() and event.content and event.content.parts:
            text = "".join(part.text or "" for part in event.content.parts if not part.thought)
            if text:
                print(f"\n--- {event.author} ---\n{text}")
        if event.error_code and not text:
            # The model stopped without an answer (e.g. safety block); explain instead of printing nothing.
            hint = STOP_REASON_HINTS.get(str(event.error_code), "Try again, or rephrase the question.")
            print(f"\n--- {event.author} ---\n(No answer: {event.error_code}. {hint})", file=sys.stderr)

    session = await session_service.get_session(app_name=app.name, user_id=USER_ID, session_id=session.id)
    if session and session.state.get(PENDING_CLARIFICATION_KEY):
        print(
            "\n(Check paused. Answer with: main.py --resume -q \"<your answer>\""
            " or send a clearer screenshot: main.py --resume <image>)"
        )

    await drain_background_tasks()


async def main() -> int:
    """Runs one CLI turn. Returns the exit code: 0 ok, 1 runtime error, 2 configuration error."""
    args = parse_args()  # exits with code 2 and a usage hint on bad arguments
    if problems := check_config():
        for problem in problems:
            print(problem.render(), file=sys.stderr)
        return 2
    try:
        await run(args)
    except Exception as exc:
        guided = explain(exc)
        log_event(
            "cli.error", level=logging.ERROR,
            error_type=type(exc).__name__, guidance=guided.title, retryable=guided.retryable,
        )
        if args.debug:
            traceback.print_exc()
        print(f"\n{guided.render()}", file=sys.stderr)
        if not args.debug:
            print("  (Run with --debug for the full traceback.)", file=sys.stderr)
        return 1
    return 0


async def drain_background_tasks(timeout: float = 15.0) -> None:
    """Waits for background tasks started during the run (e.g. Memory Bank ingestion).

    ADK sends sessions to Memory Bank as a fire-and-forget asyncio task so the
    user isn't blocked. That's ideal for a long-running server (`adk web`), but a
    CLI exits right after the response and `asyncio.run()` would cancel the task.
    Memory generation itself still runs asynchronously on the server.
    """
    pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    if pending:
        await asyncio.wait(pending, timeout=timeout)


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        exit_code = 130
    logging.shutdown()
    sys.exit(exit_code)
