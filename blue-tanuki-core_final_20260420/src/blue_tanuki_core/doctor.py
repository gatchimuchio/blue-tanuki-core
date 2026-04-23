"""Doctor: pre-flight checks before starting the app.

Checks:
- state_dir writable
- audit dir writable
- workspace writable
- security_audit.audit_settings findings
- backend healthcheck (best effort; skipped if unreachable)
"""
from __future__ import annotations

import asyncio
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .backend.base import BaseBackend
from .security_audit import AuditFinding, AuditReport, audit_settings
from .settings import Settings


DoctorStatus = Literal["ok", "warn", "fail"]


@dataclass
class DoctorReport:
    findings: list[AuditFinding] = field(default_factory=list)
    backend_health: dict[str, bool] = field(default_factory=dict)

    @property
    def status(self) -> DoctorStatus:
        if any(f.severity == "critical" for f in self.findings):
            return "fail"
        if any(f.severity == "warn" for f in self.findings):
            return "warn"
        if any(v is False for v in self.backend_health.values()):
            return "warn"
        return "ok"

    def summary(self) -> str:
        lines = [f"doctor status: {self.status.upper()}"]
        for provider, ok in self.backend_health.items():
            lines.append(f"  backend[{provider}]: {'OK' if ok else 'FAIL'}")
        if self.findings:
            from .security_audit import AuditReport as _AR
            ar = _AR(findings=list(self.findings))
            lines.append(ar.summary())
        return "\n".join(lines)


def _check_writable(path: Path, check_id: str) -> AuditFinding | None:
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path, delete=True):
            pass
        return None
    except OSError as e:
        return AuditFinding(
            check_id=check_id,
            severity="critical",
            message=f"not writable: {e}",
            fix_hint="check permissions / disk space",
            path_or_key=str(path),
        )


async def run_doctor(
    settings: Settings,
    *,
    backends: list[BaseBackend] | None = None,
    workspace: Path | None = None,
) -> DoctorReport:
    report = DoctorReport()

    # writability
    for label, p in [
        ("fs.state_dir.writable", settings.state_dir),
        ("fs.audit.writable", settings.state_dir / "audit"),
    ]:
        f = _check_writable(p, label)
        if f is not None:
            report.findings.append(f)
    if workspace is not None:
        f = _check_writable(workspace, "fs.workspace.writable")
        if f is not None:
            report.findings.append(f)

    # settings audit
    ar = audit_settings(settings)
    report.findings.extend(ar.findings)

    # backend healthcheck (best effort, bounded concurrency)
    if backends:
        async def _check(b: BaseBackend) -> tuple[str, bool]:
            try:
                ok = await asyncio.wait_for(b.healthcheck(), timeout=5)
                return b.provider, bool(ok)
            except Exception:
                return b.provider, False
        results = await asyncio.gather(*(_check(b) for b in backends))
        for provider, ok in results:
            report.backend_health[provider] = ok

    return report


__all__ = ["DoctorReport", "DoctorStatus", "run_doctor"]
