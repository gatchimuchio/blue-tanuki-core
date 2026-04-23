"""blue-tanuki CLI.

Subcommands:
  chat         Interactive REPL
  version      Print version
  verify       Verify audit hash chain for a session

Usage:
  python -m blue_tanuki_core chat [--workspace PATH] [--policy PATH] [--stub|--anthropic]
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from . import __version__
from .app import App, AppBuildOverrides
from .audit import AuditLog
from .backend.anthropic import AnthropicBackend
from .backend.stub import StubBackend
from .contracts import (
    ApprovalResponse, ClarifyResponse, ControlRequest, Message,
    new_session_id,
)
from .control_plane import ModuleRegistry
from .errors import AuditIntegrityError, ConfigError
from .gate import AllowAllGate, load_policy
from .modules.director import RuleDirector
from .modules.file_ops import FileOpsModule
from .modules.memory import MemoryModule
from .redactor import Redactor
from .settings import Settings
from .sqlite_index import SQLiteIndex


def _build_overrides(
    state_dir: Path,
    workspace: Path,
    use_stub: bool,
) -> AppBuildOverrides:
    registry = ModuleRegistry()
    registry.register(MemoryModule(state_dir / "memory.sqlite"))
    registry.register(FileOpsModule(workspace))
    backends: list = []
    if use_stub:
        backends.append(StubBackend())
    else:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ConfigError("ANTHROPIC_API_KEY", "env var is required unless --stub")
        backends.append(AnthropicBackend(api_key=key))
    return AppBuildOverrides(
        gate=AllowAllGate(),  # overridden below if policy supplied
        backends=backends,
        modules=registry,
        director=RuleDirector(),
    )


async def _cmd_chat(args: argparse.Namespace) -> int:
    state_dir = Path(args.state_dir).expanduser()
    workspace = Path(args.workspace).expanduser()
    settings = Settings(state_dir=state_dir)
    settings.ensure_dirs()

    overrides = _build_overrides(state_dir, workspace, use_stub=args.stub)
    overrides.audit_mode = args.audit_mode
    if args.policy:
        overrides.gate = load_policy(Path(args.policy))

    app = App(settings, overrides=overrides)
    await app.start()

    session_id = args.session or new_session_id()
    print(f"blue-tanuki v{__version__} | session={session_id} | workspace={workspace}")
    print("Type /help for commands. Ctrl-D or /quit to exit.\n")

    try:
        while True:
            try:
                line = input("> ").rstrip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            if line.startswith("/"):
                if line in ("/quit", "/exit"):
                    break
                if line == "/help":
                    _print_help()
                    continue
                if line == "/verify":
                    await _verify_session(app.audit, session_id)
                    continue
                if line.startswith("/session "):
                    session_id = line.split(None, 1)[1].strip() or session_id
                    print(f"session={session_id}")
                    continue
                print(f"unknown command: {line}")
                continue

            resp = await app.handle(ControlRequest(
                session_id=session_id,
                message=Message(role="user", content=line),
            ))
            await _handle_response(app, resp, session_id)
    finally:
        await app.stop()
    return 0


async def _handle_response(app: App, resp, session_id: str) -> None:
    if resp.status.code == "ok":
        for m in resp.output:
            print(f"[assistant] {m.content}")
        return
    if resp.status.code == "stopped":
        print(f"[stopped] {resp.status.reason}")
        return
    if resp.status.code == "failed":
        print(f"[failed] {resp.status.reason}")
        return
    if resp.status.code == "suspended":
        pending = resp.pending
        token = resp.status.resume_token
        if pending and pending.kind == "clarify":
            prompt = pending.clarify.prompt if pending.clarify else "clarify:"
            print(f"[clarify] {prompt}")
            ans = input("   > ").rstrip()
            resp2 = await app.resume(
                token,
                clarify=ClarifyResponse(token=token, answer=ans),
            )
            await _handle_response(app, resp2, session_id)
            return
        if pending and pending.kind == "approval":
            subject = pending.approval.subject if pending.approval else {}
            reason = pending.approval.reason if pending.approval else "?"
            print(f"[approval] {reason}")
            print(f"   subject: {subject}")
            ans = input("   approve? [y/N] ").strip().lower()
            decision = "approved" if ans in ("y", "yes") else "rejected"
            resp2 = await app.resume(
                token,
                approval=ApprovalResponse(token=token, decision=decision),
            )
            await _handle_response(app, resp2, session_id)
            return


async def _verify_session(audit: AuditLog, session_id: str) -> None:
    try:
        await audit.verify_session(session_id)
        print(f"[verify] session {session_id} chain intact")
    except AuditIntegrityError as e:
        print(f"[verify] FAILED: {e}")


def _print_help() -> None:
    print(
        "Commands:\n"
        "  /help                Show this help\n"
        "  /session <id>        Switch session id\n"
        "  /verify              Verify hash chain for current session\n"
        "  /quit, /exit         Exit\n"
        "\n"
        "Examples:\n"
        "  remember coffee is dark roast\n"
        "  recall coffee\n"
        "  what do you remember?\n"
        "  write file notes/todo.md: buy milk\n"
        "  read file notes/todo.md\n"
        "  list files\n"
    )


async def _cmd_verify(args: argparse.Namespace) -> int:
    settings = Settings(state_dir=Path(args.state_dir).expanduser())
    settings.ensure_dirs()
    index = SQLiteIndex(settings.state_dir / "audit" / "index.sqlite")
    await index.open()
    try:
        audit = AuditLog(
            root=settings.state_dir / "audit",
            redactor=Redactor(),
            index=index,
        )
        try:
            await audit.verify_session(args.session)
            print(f"OK: session {args.session} chain intact")
            return 0
        except AuditIntegrityError as e:
            print(f"FAIL: {e}", file=sys.stderr)
            return 2
    finally:
        await index.close()


async def _cmd_replay(args: argparse.Namespace) -> int:
    from .replay import replay_session
    state_dir = Path(args.state_dir).expanduser()
    if args.stub:
        backend = StubBackend()
    else:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            print("ANTHROPIC_API_KEY required (or use --stub)", file=sys.stderr)
            return 2
        backend = AnthropicBackend(api_key=key)
    report = await replay_session(
        audit_root=state_dir / "audit",
        session_id=args.session,
        backend=backend,
        require_audit_mode=not args.allow_any,
    )
    print(report.summary())
    return 0 if report.match_rate >= 1.0 or args.allow_any else 1


async def _cmd_audit(args: argparse.Namespace) -> int:
    from .security_audit import audit_settings
    settings = Settings(state_dir=Path(args.state_dir).expanduser())
    if args.policy:
        settings.policy_path = Path(args.policy).expanduser()
    report = audit_settings(settings, require_anthropic=args.anthropic)
    print(report.summary())
    if report.critical_count > 0:
        return 2
    if report.warn_count > 0 and args.strict:
        return 1
    return 0


async def _cmd_doctor(args: argparse.Namespace) -> int:
    from .doctor import run_doctor
    state_dir = Path(args.state_dir).expanduser()
    settings = Settings(state_dir=state_dir)
    if args.policy:
        settings.policy_path = Path(args.policy).expanduser()
    backends: list = []
    if args.stub:
        backends.append(StubBackend())
    elif args.anthropic:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if key:
            backends.append(AnthropicBackend(api_key=key))
    workspace = Path(args.workspace).expanduser() if args.workspace else None
    report = await run_doctor(settings, backends=backends, workspace=workspace)
    print(report.summary())
    if report.status == "fail":
        return 2
    if report.status == "warn" and args.strict:
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="blue-tanuki")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_chat = sub.add_parser("chat", help="interactive REPL")
    p_chat.add_argument("--state-dir", default="~/.blue-tanuki")
    p_chat.add_argument("--workspace", default="~/.blue-tanuki/workspace")
    p_chat.add_argument("--policy", default=None, help="path to policy.yaml")
    p_chat.add_argument("--session", default=None)
    p_chat.add_argument("--audit-mode", action="store_true",
                        help="force temperature=0 + seed (enables replay)")
    backend_grp = p_chat.add_mutually_exclusive_group()
    backend_grp.add_argument("--stub", action="store_true",
                             help="use deterministic stub backend")
    backend_grp.add_argument("--anthropic", action="store_true",
                             help="use Anthropic backend (default)")

    p_ver = sub.add_parser("version", help="print version")

    p_v = sub.add_parser("verify", help="verify session hash chain")
    p_v.add_argument("session", help="session_id (e.g. ses_abc...)")
    p_v.add_argument("--state-dir", default="~/.blue-tanuki")

    p_r = sub.add_parser("replay", help="replay LLM calls from audit and diff")
    p_r.add_argument("session", help="session_id")
    p_r.add_argument("--state-dir", default="~/.blue-tanuki")
    p_r.add_argument("--stub", action="store_true",
                     help="replay against stub backend (deterministic)")
    p_r.add_argument("--allow-any", action="store_true",
                     help="replay even non-audit-mode calls (no match guarantee)")

    p_a = sub.add_parser("audit", help="security/config audit (static)")
    p_a.add_argument("--state-dir", default="~/.blue-tanuki")
    p_a.add_argument("--policy", default=None)
    p_a.add_argument("--anthropic", action="store_true",
                     help="treat Anthropic backend as required")
    p_a.add_argument("--strict", action="store_true",
                     help="exit 1 on warn findings")

    p_d = sub.add_parser("doctor", help="pre-flight health check")
    p_d.add_argument("--state-dir", default="~/.blue-tanuki")
    p_d.add_argument("--workspace", default="~/.blue-tanuki/workspace")
    p_d.add_argument("--policy", default=None)
    p_d.add_argument("--strict", action="store_true",
                     help="exit 1 on warn findings")
    grp = p_d.add_mutually_exclusive_group()
    grp.add_argument("--stub", action="store_true")
    grp.add_argument("--anthropic", action="store_true")

    args = parser.parse_args(argv)

    if args.cmd == "version":
        print(f"blue-tanuki-core {__version__}")
        return 0
    if args.cmd == "chat":
        if not args.stub and not args.anthropic:
            args.stub = False  # default anthropic
        return asyncio.run(_cmd_chat(args))
    if args.cmd == "verify":
        return asyncio.run(_cmd_verify(args))
    if args.cmd == "replay":
        return asyncio.run(_cmd_replay(args))
    if args.cmd == "audit":
        return asyncio.run(_cmd_audit(args))
    if args.cmd == "doctor":
        return asyncio.run(_cmd_doctor(args))
    return 1


if __name__ == "__main__":
    sys.exit(main())
