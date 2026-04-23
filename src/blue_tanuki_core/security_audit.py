"""Security audit: scan state dir + settings + policy, emit checkId/severity.

Output (pydantic model):
    AuditFinding(check_id, severity, message, fix_hint, path_or_key)

severity: "info" | "warn" | "critical"

This is blue-tanuki's equivalent of `openclaw security audit`. checkIds are
stable identifiers so CI and users can grep/filter findings.

Checks implemented:
    fs.state_dir.perms_world_writable      critical
    fs.state_dir.perms_group_writable      warn
    fs.state_dir.perms_world_readable      warn
    fs.config.perms_world_writable         critical
    fs.config.perms_world_readable         warn
    fs.audit.perms_world_readable          warn
    fs.memory.perms_world_readable         warn
    policy.missing                         warn
    policy.allow_all_gate                  warn
    policy.no_rules                        info
    llm.anthropic_key_missing              warn   (only if use_anthropic is intended)
    llm.no_budget_cap                      warn
    llm.no_size_guard                      info
    audit.rotate_too_large                 info
    workspace.inside_home                  info
"""
from __future__ import annotations

import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .gate import AllowAllGate, DenyAllGate, Gate, PolicyGate, load_policy
from .settings import Settings


Severity = Literal["info", "warn", "critical"]


@dataclass
class AuditFinding:
    check_id: str
    severity: Severity
    message: str
    fix_hint: str = ""
    path_or_key: str = ""


@dataclass
class AuditReport:
    findings: list[AuditFinding] = field(default_factory=list)

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "critical")

    @property
    def warn_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "warn")

    def add(self, *args, **kw) -> None:
        self.findings.append(AuditFinding(*args, **kw))

    def summary(self) -> str:
        if not self.findings:
            return "OK: no findings"
        lines = [
            f"findings: critical={self.critical_count} "
            f"warn={self.warn_count} total={len(self.findings)}",
        ]
        for f in self.findings:
            label = {"critical": "CRIT", "warn": "WARN", "info": "INFO"}[f.severity]
            loc = f" [{f.path_or_key}]" if f.path_or_key else ""
            lines.append(f"  {label} {f.check_id}: {f.message}{loc}")
            if f.fix_hint:
                lines.append(f"       fix: {f.fix_hint}")
        return "\n".join(lines)


def _file_mode(path: Path) -> int:
    return path.stat().st_mode


def _perms(path: Path, report: AuditReport, check_prefix: str,
           key_for_message: str) -> None:
    if not path.exists():
        return
    mode = _file_mode(path)
    # world-writable
    if mode & stat.S_IWOTH:
        report.add(
            f"{check_prefix}.perms_world_writable",
            "critical",
            "world-writable",
            "chmod o-w",
            key_for_message,
        )
    if mode & stat.S_IWGRP:
        report.add(
            f"{check_prefix}.perms_group_writable",
            "warn",
            "group-writable",
            "chmod g-w",
            key_for_message,
        )
    if mode & stat.S_IROTH:
        report.add(
            f"{check_prefix}.perms_world_readable",
            "warn",
            "world-readable",
            "chmod o-r",
            key_for_message,
        )


def audit_settings(settings: Settings, *, require_anthropic: bool = False) -> AuditReport:
    report = AuditReport()

    # ── filesystem checks ─────────────────────────────────────────────
    sd = settings.state_dir
    if sd.exists():
        _perms(sd, report, "fs.state_dir", str(sd))
    audit_dir = sd / "audit"
    if audit_dir.exists():
        _perms(audit_dir, report, "fs.audit", str(audit_dir))
    mem_db = sd / "memory.sqlite"
    if mem_db.exists():
        _perms(mem_db, report, "fs.memory", str(mem_db))
    # config file (openclaw.json 相当; blue-tanuki は yaml 前提)
    for candidate in [settings.policy_path, sd / "config" / "policy.yaml"]:
        if candidate and candidate.exists():
            _perms(candidate, report, "fs.config", str(candidate))

    # ── policy checks ─────────────────────────────────────────────────
    if settings.policy_path is None:
        report.add(
            "policy.missing", "warn",
            "no policy_path configured — gate will default to AllowAllGate",
            "set BLUE_TANUKI_POLICY_PATH or pass settings.policy_path",
        )
    elif not settings.policy_path.exists():
        report.add(
            "policy.missing", "warn",
            f"policy_path does not exist: {settings.policy_path}",
            "create the file or unset policy_path",
            str(settings.policy_path),
        )
    else:
        try:
            gate = load_policy(settings.policy_path)
            if not gate.rules:
                report.add(
                    "policy.no_rules", "info",
                    "policy file has no rules",
                    "add at least one rule, or delete the policy file",
                    str(settings.policy_path),
                )
        except Exception as e:
            report.add(
                "policy.parse_error", "critical",
                f"policy failed to load: {e}",
                "fix YAML syntax or rule schema",
                str(settings.policy_path),
            )

    # ── llm / budget ──────────────────────────────────────────────────
    if require_anthropic and settings.llm.anthropic_key is None:
        report.add(
            "llm.anthropic_key_missing", "warn",
            "no anthropic_key SecretRef configured",
            "set llm.anthropic_key = SecretRef(source='env', name='ANTHROPIC_API_KEY')",
        )
    if settings.llm.max_tokens_per_turn is None and settings.llm.max_usd_per_turn is None:
        report.add(
            "llm.no_budget_cap", "warn",
            "neither max_tokens_per_turn nor max_usd_per_turn is set",
            "set at least one budget cap to avoid runaway cost",
        )
    if settings.llm.max_input_tokens > 500_000:
        report.add(
            "llm.size_guard_loose", "info",
            f"max_input_tokens={settings.llm.max_input_tokens} is very high",
            "consider lowering; runaway prompts become expensive",
        )

    # ── audit ─────────────────────────────────────────────────────────
    if settings.audit.rotate_bytes > 500 * 1024 * 1024:
        report.add(
            "audit.rotate_too_large", "info",
            "audit rotate_bytes > 500 MiB; single file will be unwieldy",
            "consider setting rotate_bytes ≤ 100 MiB",
        )

    # ── workspace placement ───────────────────────────────────────────
    home = Path.home().resolve()
    try:
        if sd.resolve().is_relative_to(home):
            pass  # expected
        else:
            report.add(
                "state_dir.outside_home", "info",
                f"state_dir outside user home: {sd}",
                "ensure this path is private to your user",
                str(sd),
            )
    except Exception:
        pass

    return report


def audit_runtime(settings: Settings, gate: Gate | None) -> AuditReport:
    """Extra checks that require a live Gate instance."""
    report = AuditReport()
    if isinstance(gate, AllowAllGate):
        report.add(
            "policy.allow_all_gate", "warn",
            "runtime gate is AllowAllGate — no inbound/tool protection",
            "load a real policy with load_policy(path)",
        )
    if isinstance(gate, DenyAllGate):
        report.add(
            "policy.deny_all_gate", "info",
            "runtime gate is DenyAllGate — every request will stop",
            "only use for testing",
        )
    return report


__all__ = [
    "AuditFinding", "AuditReport", "Severity",
    "audit_settings", "audit_runtime",
]
