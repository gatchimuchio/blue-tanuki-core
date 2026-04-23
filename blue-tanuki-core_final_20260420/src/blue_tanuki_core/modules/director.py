"""RuleDirector: lightweight rule-based intent parser.

Input: the user's latest message.
Output: DirectorPlan.

Rules (first match wins):
  "remember that <text>"      -> memory.set (namespace=notes, key=<auto>, value=<text>)
  "remember <k> is <v>"       -> memory.set (namespace=notes, key=<k>, value=<v>)
  "what do you remember"      -> memory.list
  "forget <k>"                -> memory.delete
  "recall <k>" / "what is <k>"-> memory.get (modules_only if found)
  "read file <path>"          -> file_ops.read
  "write file <path>: <text>" -> file_ops.write
  "list files"                -> file_ops.list
  otherwise                   -> llm_only
"""
from __future__ import annotations

import hashlib
import re

from ..contracts import CallContext, TypedPayload
from ..control_plane import Director, DirectorPlan, ModuleCall, ModuleRegistry
from ..pending import RunState


def _auto_key(text: str) -> str:
    return "note_" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


class RuleDirector(Director):
    async def plan(
        self, state: RunState, registry: ModuleRegistry,
    ) -> DirectorPlan:
        if not state.history:
            return DirectorPlan(action="llm_only")
        msg = state.history[-1].content.strip()
        lower = msg.lower()
        avail = set(registry.list_modules())

        # memory.list
        if lower in {"what do you remember", "what do you remember?",
                     "list notes", "show memory"}:
            if "memory" in avail:
                return DirectorPlan(
                    action="modules_only",
                    module_calls=[ModuleCall(
                        module="memory",
                        payload=TypedPayload(kind="memory.list",
                                             data={"namespace": "notes"}),
                        context=CallContext(op="list", resource="notes/*"),
                    )],
                )

        # remember <k> is <v>
        m = re.match(r"^remember\s+(?P<k>[\w\- ]{1,60})\s+is\s+(?P<v>.+)$", msg, re.IGNORECASE)
        if m and "memory" in avail:
            key = re.sub(r"\s+", "_", m.group("k").strip().lower())
            val = m.group("v").strip()
            return DirectorPlan(
                action="modules_then_llm",
                module_calls=[ModuleCall(
                    module="memory",
                    payload=TypedPayload(kind="memory.set",
                                         data={"namespace": "notes",
                                               "key": key, "value": val}),
                    context=CallContext(op="set", resource=f"notes/{key}"),
                )],
            )

        # remember that <text>
        m = re.match(r"^remember\s+that\s+(?P<t>.+)$", msg, re.IGNORECASE)
        if m and "memory" in avail:
            text = m.group("t").strip()
            key = _auto_key(text)
            return DirectorPlan(
                action="modules_then_llm",
                module_calls=[ModuleCall(
                    module="memory",
                    payload=TypedPayload(kind="memory.set",
                                         data={"namespace": "notes",
                                               "key": key, "value": text}),
                    context=CallContext(op="set", resource=f"notes/{key}"),
                )],
            )

        # forget <k>
        m = re.match(r"^forget\s+(?P<k>[\w\- ]{1,60})$", msg, re.IGNORECASE)
        if m and "memory" in avail:
            key = re.sub(r"\s+", "_", m.group("k").strip().lower())
            return DirectorPlan(
                action="modules_then_llm",
                module_calls=[ModuleCall(
                    module="memory",
                    payload=TypedPayload(kind="memory.delete",
                                         data={"namespace": "notes", "key": key}),
                    context=CallContext(op="delete", resource=f"notes/{key}"),
                )],
            )

        # recall <k> / what is <k>
        m = re.match(r"^(?:recall|what\s+is)\s+(?P<k>[\w\- ]{1,60})\??$", msg, re.IGNORECASE)
        if m and "memory" in avail:
            key = re.sub(r"\s+", "_", m.group("k").strip().lower())
            return DirectorPlan(
                action="modules_then_llm",
                module_calls=[ModuleCall(
                    module="memory",
                    payload=TypedPayload(kind="memory.get",
                                         data={"namespace": "notes", "key": key}),
                    context=CallContext(op="get", resource=f"notes/{key}"),
                )],
            )

        # read file <path>
        m = re.match(r"^read\s+file\s+(?P<p>\S+)$", msg, re.IGNORECASE)
        if m and "file_ops" in avail:
            path = m.group("p")
            return DirectorPlan(
                action="modules_then_llm",
                module_calls=[ModuleCall(
                    module="file_ops",
                    payload=TypedPayload(kind="file_ops.read", data={"path": path}),
                    context=CallContext(op="read", resource=path),
                )],
            )

        # write file <path>: <text>
        m = re.match(r"^write\s+file\s+(?P<p>\S+)\s*[:：]\s*(?P<t>.+)$", msg, re.IGNORECASE | re.DOTALL)
        if m and "file_ops" in avail:
            path = m.group("p")
            text = m.group("t")
            return DirectorPlan(
                action="modules_then_llm",
                module_calls=[ModuleCall(
                    module="file_ops",
                    payload=TypedPayload(kind="file_ops.write",
                                         data={"path": path, "content": text}),
                    context=CallContext(op="write", resource=path),
                )],
            )

        # list files [<path>]
        m = re.match(r"^list\s+files(?:\s+(?P<p>\S+))?$", msg, re.IGNORECASE)
        if m and "file_ops" in avail:
            path = m.group("p") or "."
            return DirectorPlan(
                action="modules_then_llm",
                module_calls=[ModuleCall(
                    module="file_ops",
                    payload=TypedPayload(kind="file_ops.list", data={"path": path}),
                    context=CallContext(op="list", resource=path),
                )],
            )

        # fetch <url>
        m = re.match(
            r"^(?:fetch|get)\s+(?:url\s+)?(?P<u>https?://\S+)$",
            msg, re.IGNORECASE,
        )
        if m and "web_fetch" in avail:
            url = m.group("u")
            return DirectorPlan(
                action="modules_then_llm",
                module_calls=[ModuleCall(
                    module="web_fetch",
                    payload=TypedPayload(kind="web_fetch.get", data={"url": url}),
                    context=CallContext(op="get", resource=url),
                )],
            )

        return DirectorPlan(action="llm_only")


__all__ = ["RuleDirector"]
