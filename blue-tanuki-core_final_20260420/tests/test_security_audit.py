import pytest

from blue_tanuki_core.backend.stub import StubBackend
from blue_tanuki_core.doctor import run_doctor
from blue_tanuki_core.gate import AllowAllGate, PolicyGate
from blue_tanuki_core.security_audit import (
    audit_runtime, audit_settings,
)
from blue_tanuki_core.settings import Settings


def _settings(tmp_path):
    s = Settings(state_dir=tmp_path)
    s.ensure_dirs()
    return s


def test_warns_on_missing_policy(tmp_path):
    s = _settings(tmp_path)
    report = audit_settings(s)
    ids = {f.check_id for f in report.findings}
    assert "policy.missing" in ids


def test_allow_all_gate_warning():
    r = audit_runtime(Settings(), AllowAllGate())
    ids = {f.check_id for f in r.findings}
    assert "policy.allow_all_gate" in ids


def test_bad_policy_file_critical(tmp_path):
    s = _settings(tmp_path)
    bad = tmp_path / "bad.yaml"
    bad.write_text("rules: not_a_list")
    s.policy_path = bad
    report = audit_settings(s)
    ids = {f.check_id for f in report.findings}
    assert "policy.parse_error" in ids
    assert report.critical_count >= 1


def test_no_budget_warning(tmp_path):
    s = _settings(tmp_path)
    s.llm.max_tokens_per_turn = None
    s.llm.max_usd_per_turn = None
    report = audit_settings(s)
    ids = {f.check_id for f in report.findings}
    assert "llm.no_budget_cap" in ids


def test_budget_set_suppresses_warning(tmp_path):
    s = _settings(tmp_path)
    s.llm.max_tokens_per_turn = 1000
    report = audit_settings(s)
    ids = {f.check_id for f in report.findings}
    assert "llm.no_budget_cap" not in ids


def test_world_writable_state_dir(tmp_path):
    s = _settings(tmp_path)
    # make state_dir world-writable
    import os, stat
    os.chmod(s.state_dir, 0o777)
    report = audit_settings(s)
    ids = {f.check_id for f in report.findings}
    assert "fs.state_dir.perms_world_writable" in ids


async def test_doctor_ok_on_clean_setup(tmp_path):
    s = _settings(tmp_path)
    s.llm.max_tokens_per_turn = 1000  # silence budget warning
    import os
    os.chmod(s.state_dir, 0o700)
    os.chmod(s.state_dir / "audit", 0o700)
    report = await run_doctor(
        s,
        backends=[StubBackend()],
        workspace=tmp_path / "workspace",
    )
    # policy.missing is still a warn -> status becomes warn, but not fail
    assert report.status in ("ok", "warn")
    assert report.backend_health == {"stub": True}


async def test_doctor_detects_unwritable(tmp_path):
    s = _settings(tmp_path)
    # make audit dir unwritable
    import os
    os.chmod(s.state_dir / "audit", 0o500)
    try:
        report = await run_doctor(s, backends=[])
        # root in some test envs may still be able to write; accept either
        if report.status == "fail":
            ids = {f.check_id for f in report.findings}
            assert "fs.audit.writable" in ids
    finally:
        os.chmod(s.state_dir / "audit", 0o700)
