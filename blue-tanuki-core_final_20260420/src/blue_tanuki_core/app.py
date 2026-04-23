"""App lifecycle: start, handle/resume, stop.

Single-process bootstrap tying Settings -> AuditLog -> Gate -> LLMPipe -> ControlPlane.
Default wiring now boots a usable upstream-control core: built-in director,
built-in modules, and restart-safe pending persistence.
"""
from __future__ import annotations

import asyncio
import signal
from importlib.resources import files
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .audit import AuditLog
from .backend.base import BaseBackend
from .backend.stub import StubBackend
from .contracts import ApprovalResponse, ClarifyResponse, ControlRequest, ControlResponse
from .control_plane import ControlPlane, ControlPlaneConfig, Director, ModuleRegistry
from .gate import Gate, load_policy
from .modules.director import RuleDirector
from .modules.file_ops import FileOpsModule
from .modules.memory import MemoryModule
from .modules.web_fetch import WebFetchModule
from .pending import FilePendingStore, PendingStore
from .pipe import BudgetGuard, LLMPipe, NullTelemetry, PromptSizeGuard
from .redactor import Redactor
from .settings import Settings
from .sqlite_index import SQLiteIndex
from .errors import ConfigError


@dataclass
class AppBuildOverrides:
    """Hooks to inject custom components (tests, experiments)."""
    gate: Gate | None = None
    backends: list[BaseBackend] | None = None
    modules: ModuleRegistry | None = None
    director: Director | None = None
    audit_mode: bool = False


class App:
    def __init__(self, settings: Settings, overrides: AppBuildOverrides | None = None):
        self.settings = settings
        self.overrides = overrides or AppBuildOverrides()
        self._started = False
        self._stopping = False
        self._inflight: set[asyncio.Task[Any]] = set()
        self._sweeper_task: asyncio.Task[Any] | None = None

        self.index: SQLiteIndex | None = None
        self.audit: AuditLog | None = None
        self.gate: Gate | None = None
        self.pipe: LLMPipe | None = None
        self.modules: ModuleRegistry | None = None
        self.pending: PendingStore | None = None
        self.control: ControlPlane | None = None

    def _build_default_registry(self) -> ModuleRegistry:
        s = self.settings
        registry = ModuleRegistry()
        registry.register(MemoryModule(s.state_dir / "memory.sqlite"))
        registry.register(FileOpsModule(s.state_dir / "workspace"))
        registry.register(WebFetchModule())
        return registry

    async def start(self) -> None:
        if self._started:
            return
        s = self.settings
        s.ensure_dirs()

        self.index = SQLiteIndex(s.state_dir / "audit" / "index.sqlite")
        await self.index.open()

        redactor = Redactor(
            secret_keys=set(s.audit.redact_keys),
            patterns=s.audit.redact_patterns,
        )
        self.audit = AuditLog(
            root=s.state_dir / "audit",
            redactor=redactor,
            index=self.index,
            rotate_bytes=s.audit.rotate_bytes,
        )

        if self.overrides.gate is not None:
            self.gate = self.overrides.gate
        elif s.policy_path is not None:
            self.gate = load_policy(s.policy_path)
        else:
            default_policy = files("blue_tanuki_core").joinpath("policy.default.yaml")
            self.gate = load_policy(Path(str(default_policy)))

        backends: list[BaseBackend] = self.overrides.backends or [StubBackend()]
        if s.llm.anthropic_key is not None:
            s.llm.anthropic_key.resolve()
        if s.llm.openai_key is not None:
            s.llm.openai_key.resolve()
        self.pipe = LLMPipe(
            backends=backends,
            budget=BudgetGuard(
                max_tokens_per_turn=s.llm.max_tokens_per_turn,
                max_usd_per_turn=s.llm.max_usd_per_turn,
                usd_per_1k_prompt=s.llm.usd_per_1k_prompt,
                usd_per_1k_completion=s.llm.usd_per_1k_completion,
            ),
            size_guard=PromptSizeGuard(max_input_tokens=s.llm.max_input_tokens),
            telemetry=NullTelemetry(),
            default_timeout_s=s.llm.timeout_s,
        )

        self.modules = self.overrides.modules or self._build_default_registry()
        self.pending = FilePendingStore(s.state_dir / "pending")

        self.control = ControlPlane(
            gate=self.gate,
            audit=self.audit,
            pipe=self.pipe,
            modules=self.modules,
            pending_store=self.pending,
            config=ControlPlaneConfig(
                pending_ttl_s=s.runtime.pending_ttl_s,
                module_timeout_s=s.runtime.module_timeout_s,
                llm_default_model=s.llm.default_model,
                llm_audit_model=s.llm.audit_model,
                llm_audit_seed=s.llm.audit_seed,
                audit_mode_default=self.overrides.audit_mode,
            ),
            director=self.overrides.director or RuleDirector(),
        )

        self._sweeper_task = asyncio.create_task(self._pending_sweeper())
        self._started = True

    async def handle(self, req: ControlRequest) -> ControlResponse:
        if not self._started or self._stopping:
            raise ConfigError("app", "not started or stopping")
        assert self.control is not None
        task = asyncio.create_task(self.control.handle(req))
        self._inflight.add(task)
        try:
            return await task
        finally:
            self._inflight.discard(task)

    async def resume(
        self,
        token: str,
        *,
        approval: ApprovalResponse | None = None,
        clarify: ClarifyResponse | None = None,
    ) -> ControlResponse:
        if not self._started or self._stopping:
            raise ConfigError("app", "not started or stopping")
        assert self.control is not None
        return await self.control.resume(token, approval=approval, clarify=clarify)

    async def cancel(self, token: str) -> None:
        if not self._started:
            return
        assert self.control is not None
        await self.control.cancel(token)

    async def stop(self, *, grace_s: float | None = None) -> None:
        if not self._started or self._stopping:
            return
        self._stopping = True
        grace = grace_s if grace_s is not None else self.settings.runtime.grace_shutdown_s

        if self._sweeper_task is not None:
            self._sweeper_task.cancel()
            try:
                await self._sweeper_task
            except (asyncio.CancelledError, Exception):
                pass
            self._sweeper_task = None

        if self._inflight:
            try:
                await asyncio.wait_for(asyncio.gather(*self._inflight, return_exceptions=True), timeout=grace)
            except asyncio.TimeoutError:
                for t in self._inflight:
                    if not t.done():
                        t.cancel()

        if self.audit is not None:
            await self.audit.close()
        if self.index is not None:
            await self.index.close()

        self._started = False

    def install_signal_handlers(self, loop: asyncio.AbstractEventLoop) -> None:
        def _handler(_sig, _frame):
            asyncio.run_coroutine_threadsafe(self.stop(), loop)
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(self.stop()))
            except (NotImplementedError, RuntimeError):
                signal.signal(sig, _handler)

    async def _pending_sweeper(self) -> None:
        assert self.pending is not None and self.audit is not None
        try:
            while not self._stopping:
                await asyncio.sleep(30)
                expired = await self.pending.sweep_expired()
                for tok in expired:
                    try:
                        await self.audit.record("suspend_expired", session_id="unknown", payload={"token": tok})
                    except Exception:
                        pass
        except asyncio.CancelledError:
            return


__all__ = ["App", "AppBuildOverrides"]
