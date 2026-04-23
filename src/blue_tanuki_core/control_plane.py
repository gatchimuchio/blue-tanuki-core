"""ControlPlane: state machine driving a turn.

Scope of this skeleton:
- handle(req): gate(inbound) -> director plan -> module dispatch loop -> LLM -> response
- director is pluggable; if unset, behavior falls back to "LLM only"
- resume(token, ...): restore RunState, continue
- cancel(token): drop pending entry

Everything that crosses layer boundaries goes through audit.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Protocol

from .audit import AuditLog
from .backend.base import LLMCallRequest, LLMCallResponse
from .contracts import (
    ApprovalRequest, ApprovalResponse, CallContext, ClarifyRequest, ClarifyResponse,
    ControlRequest, ControlResponse, ControlStatus,
    GateDecision, GateSubject,
    Message, ModuleRequest, ModuleResponse, Pending, TypedPayload,
    canonical_json_bytes, new_call_id, new_resume_token, new_request_id, payload_hash,
)
from .errors import (
    BlueTanukiError, GateStopSignal, GateSuspendSignal,
    ModuleCrash, ModuleNotFound, ModuleRefused, ModuleTimeout,
)
from .gate import Gate
from .pending import InMemoryPendingStore, PendingStore, RunState
from .pipe import LLMPipe


# ──────────────────────────────────────────────────────────────────────
# Module registry (abstract; module bodies are out of scope)
# ──────────────────────────────────────────────────────────────────────

class Module:
    name: str = ""

    async def handle(self, req: ModuleRequest) -> ModuleResponse: ...


class ModuleRegistry:
    def __init__(self) -> None:
        self._modules: dict[str, Module] = {}

    def register(self, module: Module) -> None:
        if not module.name:
            raise ValueError("module.name must be set")
        self._modules[module.name] = module

    def get(self, name: str) -> Module:
        if name not in self._modules:
            raise ModuleNotFound(name)
        return self._modules[name]

    def list_modules(self) -> list[str]:
        return sorted(self._modules.keys())


# ──────────────────────────────────────────────────────────────────────
# Director abstraction
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ModuleCall:
    """One planned module call."""
    module: str
    payload: TypedPayload
    context: CallContext


@dataclass
class DirectorPlan:
    """Plan produced by a Director for a single turn.

    action:
      - "modules_then_llm": run module_calls in order, then call LLM with results folded in
      - "llm_only": skip modules and call LLM directly
      - "modules_only": run module_calls, return the last tool result verbatim as assistant text
      - "reply_direct": skip both module and LLM; return direct_reply immediately
    """
    action: str = "llm_only"
    module_calls: list[ModuleCall] | None = None
    direct_reply: str | None = None


class Director(Protocol):
    async def plan(
        self, state: RunState, registry: "ModuleRegistry",
    ) -> DirectorPlan: ...


class NullDirector:
    """Default director: always LLM-only."""

    async def plan(self, state: RunState, registry: ModuleRegistry) -> DirectorPlan:
        return DirectorPlan(action="llm_only")


def _hydrate_plan(raw: Any) -> DirectorPlan | None:
    if raw is None:
        return None
    if isinstance(raw, DirectorPlan):
        return raw
    if not isinstance(raw, dict):
        return None
    calls_raw = raw.get("module_calls") or []
    calls: list[ModuleCall] = []
    for item in calls_raw:
        if isinstance(item, ModuleCall):
            calls.append(item)
            continue
        if not isinstance(item, dict):
            continue
        calls.append(ModuleCall(
            module=item["module"],
            payload=TypedPayload.model_validate(item["payload"]),
            context=CallContext.model_validate(item["context"]),
        ))
    return DirectorPlan(
        action=raw.get("action", "llm_only"),
        module_calls=calls,
        direct_reply=raw.get("direct_reply"),
    )


# ──────────────────────────────────────────────────────────────────────
# ControlPlane
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ControlPlaneConfig:
    pending_ttl_s: int = 3600
    module_timeout_s: float = 30
    audit_mode_default: bool = False
    llm_default_model: str = "claude-sonnet-4-20250514"
    llm_audit_model: str = "claude-sonnet-4-20250514"
    llm_audit_seed: int = 1


class ControlPlane:
    def __init__(
        self,
        gate: Gate,
        audit: AuditLog,
        pipe: LLMPipe,
        modules: ModuleRegistry,
        pending_store: PendingStore | None = None,
        config: ControlPlaneConfig | None = None,
        director: Director | None = None,
    ):
        self.gate = gate
        self.audit = audit
        self.pipe = pipe
        self.modules = modules
        self.pending = pending_store or InMemoryPendingStore()
        self.config = config or ControlPlaneConfig()
        self.director = director or NullDirector()

    # ── public API ───────────────────────────────────────────────────

    async def handle(self, req: ControlRequest) -> ControlResponse:
        await self.audit.open_session(req.session_id)
        await self.audit.record(
            "request_received",
            session_id=req.session_id,
            request_id=req.request_id,
            turn_id=req.turn_id,
            payload={"message": req.message.model_dump()},
        )
        state = RunState(
            request_id=req.request_id,
            session_id=req.session_id,
            turn_id=req.turn_id,
            phase="inbound",
            history=[req.message],
            audit_mode=self.config.audit_mode_default,
        )
        return await self._run_protected(req, state)

    async def resume(
        self,
        token: str,
        *,
        approval: ApprovalResponse | None = None,
        clarify: ClarifyResponse | None = None,
    ) -> ControlResponse:
        pending, state = await self.pending.take(token)
        parent_request_id = state.request_id
        state.request_id = new_request_id()
        req = ControlRequest(
            request_id=state.request_id,
            session_id=state.session_id,
            turn_id=state.turn_id,
            message=state.history[-1] if state.history else Message(role="system", content=""),
        )
        if pending.kind == "approval":
            decision = approval.decision if approval else "rejected"
            await self.audit.record(
                "resume",
                session_id=state.session_id,
                request_id=state.request_id,
                turn_id=state.turn_id,
                payload={
                    "kind": "approval",
                    "decision": decision,
                    "note": approval.note if approval else None,
                    "token": token,
                    "parent_request_id": parent_request_id,
                },
            )
            if decision != "approved":
                await self.audit.record(
                    "resume_rejected",
                    session_id=state.session_id,
                    request_id=state.request_id,
                    turn_id=state.turn_id,
                    payload={"token": token, "parent_request_id": parent_request_id},
                )
                return self._stopped(req, "approval rejected")
            state.approval_granted = True
            suspended_direction = state.extra.pop("last_suspend_direction", None)
            if suspended_direction:
                state.approval_bypass_directions.append(suspended_direction)
        else:
            if clarify is None:
                await self.audit.record(
                    "resume_rejected",
                    session_id=state.session_id,
                    request_id=state.request_id,
                    turn_id=state.turn_id,
                    payload={"token": token, "kind": "clarify", "parent_request_id": parent_request_id},
                )
                return self._stopped(req, "clarify cancelled")
            await self.audit.record(
                "resume",
                session_id=state.session_id,
                request_id=state.request_id,
                turn_id=state.turn_id,
                payload={
                    "kind": "clarify",
                    "answer_preview": clarify.answer[:80],
                    "token": token,
                    "parent_request_id": parent_request_id,
                },
            )
            state.history.append(Message(role="user", content=clarify.answer))
        return await self._run_protected(req, state)

    async def cancel(self, token: str) -> None:
        await self.pending.cancel(token)

    # ── core loop ────────────────────────────────────────────────────

    async def _run_protected(
        self, req: ControlRequest, state: RunState,
    ) -> ControlResponse:
        try:
            return await self._run(req, state)
        except GateSuspendSignal as e:
            return self._suspended_from_signal(req, state, e)
        except GateStopSignal as e:
            await self.audit.record(
                "typed_error",
                session_id=state.session_id,
                request_id=state.request_id,
                turn_id=state.turn_id,
                payload={"type": "GateStopSignal", "reason": e.decision.reason,
                         "matched": e.decision.matched_rule_ids},
            )
            return self._stopped(req, e.decision.reason or "gate stop")
        except BlueTanukiError as e:
            await self.audit.record(
                "typed_error",
                session_id=state.session_id,
                request_id=state.request_id,
                turn_id=state.turn_id,
                payload={"type": type(e).__name__, "detail": str(e)},
            )
            return self._stopped(req, str(e))
        except BaseException as e:
            await self.audit.record_exception(req, e, session_id=state.session_id)
            return self._failed(req, "internal failure")

    async def _run(self, req: ControlRequest, state: RunState) -> ControlResponse:
        # Hook 1: inbound
        if state.phase == "inbound":
            await self._run_gate(
                GateSubject(
                    direction="inbound",
                    session_id=state.session_id,
                    request_id=state.request_id,
                    turn_id=state.turn_id,
                    caller="user",
                    payload_hash=payload_hash(state.history[-1].model_dump()),
                    payload_size=len(canonical_json_bytes(state.history[-1].model_dump())),
                    content=state.history[-1].content,
                    messages_preview=[state.history[-1].content[:200]],
                ),
                state,
            )
            state.phase = "plan"

        # Director plan
        if state.phase == "plan":
            plan = await self.director.plan(state, self.modules)
            state.extra["plan"] = {
                "action": plan.action,
                "modules": [c.module for c in (plan.module_calls or [])],
            }
            state.extra["plan_obj"] = plan
            state.phase = "dispatch"

        plan = _hydrate_plan(state.extra.get("plan_obj"))
        assert plan is not None
        state.extra["plan_obj"] = plan

        # Direct reply: bypass LLM entirely
        if plan.action == "reply_direct":
            out_msg = Message(role="assistant", content=plan.direct_reply or "")
            state.history.append(out_msg)
            await self.audit.record(
                "response_sent",
                session_id=state.session_id,
                request_id=state.request_id,
                turn_id=state.turn_id,
                payload={"message": out_msg.model_dump(), "via": "direct_reply"},
            )
            return ControlResponse(
                request_id=state.request_id,
                status=ControlStatus(code="ok"),
                output=[out_msg],
            )

        # Module dispatch loop
        if state.phase == "dispatch" and plan.module_calls:
            module_results: list[tuple[ModuleCall, ModuleResponse]] = []
            parent_call_id = state.extra.get("turn_parent_call_id")
            for call in plan.module_calls:
                mreq = self.make_module_request(
                    state=state,
                    module=call.module,
                    payload=call.payload,
                    context=call.context,
                    caller="director",
                    parent_call_id=parent_call_id,
                )
                mresp = await self.dispatch_module(mreq, state=state)
                module_results.append((call, mresp))
            state.extra["module_results"] = module_results

        # modules_only: synthesize a reply from the last module result
        if plan.action == "modules_only":
            results = state.extra.get("module_results") or []
            if results:
                _, last = results[-1]
                text = (
                    last.result.get("text")
                    or last.result.get("value")
                    or str(last.result)
                )
            else:
                text = ""
            out_msg = Message(role="assistant", content=text)
            state.history.append(out_msg)
            await self.audit.record(
                "response_sent",
                session_id=state.session_id,
                request_id=state.request_id,
                turn_id=state.turn_id,
                payload={"message": out_msg.model_dump(), "via": "modules_only"},
            )
            return ControlResponse(
                request_id=state.request_id,
                status=ControlStatus(code="ok"),
                output=[out_msg],
            )

        # Fold module results into history as system/tool messages before LLM
        results = state.extra.pop("module_results", None) or []
        if results:
            summary_lines = []
            for call, mresp in results:
                summary_lines.append(
                    f"[tool:{call.module} op={call.context.op} "
                    f"status={mresp.status.code}] {mresp.result}"
                )
            state.history.append(Message(
                role="system",
                content="Tool results:\n" + "\n".join(summary_lines),
            ))

        # Hook 4: pre_llm
        llm_req = self._build_llm_request(state)
        state.pending_llm = llm_req
        await self._run_gate(
            GateSubject(
                direction="pre_llm",
                session_id=state.session_id,
                request_id=state.request_id,
                turn_id=state.turn_id,
                caller="director",
                model=llm_req.model,
                messages_preview=[m.content[:200] for m in llm_req.messages],
                payload_hash=payload_hash(llm_req.model_dump()),
                payload_size=len(canonical_json_bytes(llm_req.model_dump())),
            ),
            state,
        )

        # LLM call
        call_id = new_call_id()
        await self.audit.record(
            "llm_request",
            session_id=state.session_id,
            request_id=state.request_id,
            turn_id=state.turn_id,
            call_id=call_id,
            payload={"model": llm_req.model, "messages": [m.model_dump() for m in llm_req.messages],
                     "temperature": llm_req.temperature, "seed": llm_req.seed},
        )
        resp = await self.pipe.call(llm_req, audit_mode=state.audit_mode)
        await self.audit.record(
            "llm_response",
            session_id=state.session_id,
            request_id=state.request_id,
            turn_id=state.turn_id,
            call_id=call_id,
            payload={"provider": resp.provider, "model": resp.model,
                     "content": resp.content, "finish_reason": resp.finish_reason,
                     "usage": {"prompt": resp.usage_prompt_tokens,
                               "completion": resp.usage_completion_tokens},
                     "latency_ms": resp.latency_ms},
        )

        # Hook 5: post_llm
        await self._run_gate(
            GateSubject(
                direction="post_llm",
                session_id=state.session_id,
                request_id=state.request_id,
                turn_id=state.turn_id,
                caller="llm",
                model=resp.model,
                finish_reason=resp.finish_reason,
                messages_preview=[resp.content[:200]],
                payload_hash=payload_hash(resp.model_dump()),
                payload_size=len(canonical_json_bytes(resp.model_dump())),
            ),
            state,
        )

        out_msg = Message(role="assistant", content=resp.content)
        state.history.append(out_msg)

        await self.audit.record(
            "response_sent",
            session_id=state.session_id,
            request_id=state.request_id,
            turn_id=state.turn_id,
            payload={"message": out_msg.model_dump()},
        )
        return ControlResponse(
            request_id=state.request_id,
            status=ControlStatus(code="ok"),
            output=[out_msg],
        )

    # ── module dispatch (skeleton; used by director + public API) ──

    async def dispatch_module(
        self,
        mreq: ModuleRequest,
        state: RunState | None = None,
    ) -> ModuleResponse:
        # Hook 2: pre_module
        await self._run_gate(
            GateSubject(
                direction="pre_module",
                session_id=mreq.session_id,
                request_id=mreq.request_id,
                turn_id=mreq.turn_id,
                caller=mreq.caller,
                module=mreq.module,
                op=mreq.context.op,
                resource=mreq.context.resource,
                payload_hash=payload_hash(mreq.payload.model_dump()),
                payload_size=len(canonical_json_bytes(mreq.payload.model_dump())),
            ),
            state=state,
        )
        await self.audit.record(
            "module_request",
            session_id=mreq.session_id,
            request_id=mreq.request_id,
            turn_id=mreq.turn_id,
            call_id=mreq.call_id,
            parent_call_id=mreq.parent_call_id,
            payload={"module": mreq.module, "payload": mreq.payload.model_dump(),
                     "context": mreq.context.model_dump(), "caller": mreq.caller},
        )
        module = self.modules.get(mreq.module)
        try:
            mresp = await asyncio.wait_for(
                module.handle(mreq),
                timeout=self.config.module_timeout_s,
            )
        except asyncio.TimeoutError as e:
            raise ModuleTimeout(mreq.module, self.config.module_timeout_s) from e
        except ModuleRefused:
            raise
        except Exception as e:
            raise ModuleCrash(mreq.module, str(e)) from e

        # Hook 3: post_module
        total_out = sum(se.bytes_out for se in mresp.side_effects)
        await self._run_gate(
            GateSubject(
                direction="post_module",
                session_id=mreq.session_id,
                request_id=mreq.request_id,
                turn_id=mreq.turn_id,
                caller=mreq.caller,
                module=mreq.module,
                op=mreq.context.op,
                resource=mreq.context.resource,
                payload_hash=payload_hash(mresp.model_dump()),
                side_effects_preview=[se.model_dump() for se in mresp.side_effects],
                side_effect_bytes_out_total=total_out,
            ),
            state=state,
        )
        await self.audit.record(
            "module_response",
            session_id=mreq.session_id,
            request_id=mreq.request_id,
            turn_id=mreq.turn_id,
            call_id=mreq.call_id,
            parent_call_id=mreq.parent_call_id,
            payload={"status": mresp.status.model_dump(),
                     "result_keys": sorted(mresp.result.keys()),
                     "side_effects": [se.model_dump() for se in mresp.side_effects],
                     "duration_ms": mresp.duration_ms},
        )
        return mresp

    def make_module_request(
        self,
        *,
        state: RunState,
        module: str,
        payload: TypedPayload,
        context: CallContext,
        caller: str = "director",
        parent_call_id: str | None = None,
    ) -> ModuleRequest:
        """High-level constructor that prevents context-drop bugs."""
        return ModuleRequest(
            call_id=new_call_id(),
            parent_call_id=parent_call_id,
            request_id=state.request_id,
            session_id=state.session_id,
            turn_id=state.turn_id,
            module=module,
            payload=payload,
            context=context,
            caller=caller,  # type: ignore[arg-type]
        )

    # ── internals ────────────────────────────────────────────────────

    async def _run_gate(self, subject: GateSubject, state: RunState | None) -> None:
        # one-shot approval bypass: if a prior resume(approved) marked this
        # direction as bypassable, consume it and pass without consulting the gate.
        if state is not None and subject.direction in state.approval_bypass_directions:
            state.approval_bypass_directions.remove(subject.direction)
            await self.audit.record(
                "gate_decision",
                session_id=subject.session_id,
                request_id=subject.request_id,
                turn_id=subject.turn_id,
                payload={
                    "direction": subject.direction,
                    "action": "pass",
                    "reason": "approval_bypass",
                    "policy_id": "approval_bypass",
                    "matched": [],
                    "module": subject.module,
                    "op": subject.op,
                    "resource": subject.resource,
                },
            )
            return
        decision = self.gate.evaluate(subject)
        await self.audit.record(
            "gate_decision",
            session_id=subject.session_id,
            request_id=subject.request_id,
            turn_id=subject.turn_id,
            payload={
                "direction": subject.direction,
                "action": decision.action,
                "reason": decision.reason,
                "policy_id": decision.policy_id,
                "matched": decision.matched_rule_ids,
                "module": subject.module,
                "op": subject.op,
                "resource": subject.resource,
            },
        )
        if decision.action == "pass":
            return
        if decision.action == "stop":
            raise GateStopSignal(decision)
        if decision.action == "suspend_approval":
            if state is None:
                # module-level suspension without RunState is unsupported in this skeleton
                raise GateStopSignal(GateDecision(
                    action="stop",
                    reason="suspend_approval not supported outside RunState",
                ))
            token = new_resume_token()
            approval = ApprovalRequest(
                token=token,
                subject=decision.approval_subject or {
                    "direction": subject.direction,
                    "module": subject.module,
                    "op": subject.op,
                    "resource": subject.resource,
                },
                reason=decision.reason,
            )
            pending_obj = Pending(kind="approval", token=token, approval=approval)
            state.extra["last_resume_token"] = token
            state.extra["last_pending"] = pending_obj.model_dump()
            state.extra["last_suspend_direction"] = subject.direction
            await self.pending.put(
                token, pending_obj, state, ttl_s=self.config.pending_ttl_s,
            )
            await self.audit.record(
                "suspend",
                session_id=state.session_id,
                request_id=state.request_id,
                turn_id=state.turn_id,
                payload={"kind": "approval", "token": token, "reason": decision.reason},
            )
            raise GateSuspendSignal(decision, kind="approval")
        if decision.action == "suspend_clarify":
            if state is None:
                raise GateStopSignal(GateDecision(
                    action="stop",
                    reason="suspend_clarify not supported outside RunState",
                ))
            token = new_resume_token()
            clarify = ClarifyRequest(
                token=token,
                prompt=decision.clarify_prompt or "Please clarify.",
            )
            pending_obj = Pending(kind="clarify", token=token, clarify=clarify)
            state.extra["last_resume_token"] = token
            state.extra["last_pending"] = pending_obj.model_dump()
            await self.pending.put(
                token, pending_obj, state, ttl_s=self.config.pending_ttl_s,
            )
            await self.audit.record(
                "suspend",
                session_id=state.session_id,
                request_id=state.request_id,
                turn_id=state.turn_id,
                payload={"kind": "clarify", "token": token,
                         "prompt": clarify.prompt},
            )
            raise GateSuspendSignal(decision, kind="clarify")

    def _build_llm_request(self, state: RunState) -> LLMCallRequest:
        if state.audit_mode:
            return LLMCallRequest(
                model=self.config.llm_audit_model,
                messages=list(state.history),
                max_tokens=1024,
                temperature=0,
                seed=self.config.llm_audit_seed,
            )
        return LLMCallRequest(
            model=self.config.llm_default_model,
            messages=list(state.history),
            max_tokens=1024,
        )

    # ── response builders ────────────────────────────────────────────

    def _stopped(self, req: ControlRequest, reason: str) -> ControlResponse:
        return ControlResponse(
            request_id=req.request_id,
            status=ControlStatus(code="stopped", reason=reason),
        )

    def _failed(self, req: ControlRequest, reason: str) -> ControlResponse:
        return ControlResponse(
            request_id=req.request_id,
            status=ControlStatus(code="failed", reason=reason),
        )

    def _suspended_from_signal(
        self, req: ControlRequest, state: RunState, sig: GateSuspendSignal,
    ) -> ControlResponse:
        token = state.extra.get("last_resume_token")
        if token is None:
            return self._stopped(req, "suspend without token (bug)")
        pending_dump = state.extra.get("last_pending")
        pending = (
            Pending.model_validate(pending_dump)
            if pending_dump is not None
            else Pending(kind="approval" if sig.kind == "approval" else "clarify", token=token)
        )
        return ControlResponse(
            request_id=req.request_id,
            status=ControlStatus(code="suspended", resume_token=token),
            pending=pending,
        )


__all__ = [
    "ControlPlane", "ControlPlaneConfig",
    "Module", "ModuleRegistry",
    "Director", "NullDirector", "DirectorPlan", "ModuleCall",
]
