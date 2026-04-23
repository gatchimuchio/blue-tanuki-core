import pytest

from blue_tanuki_core.audit import AuditLog
from blue_tanuki_core.errors import AuditIntegrityError
from blue_tanuki_core.redactor import Redactor
from blue_tanuki_core.sqlite_index import SQLiteIndex


@pytest.fixture
async def audit(tmp_path):
    idx = SQLiteIndex(tmp_path / "index.sqlite")
    await idx.open()
    a = AuditLog(
        root=tmp_path / "audit",
        redactor=Redactor(secret_keys={"token", "api_key"}),
        index=idx,
    )
    yield a
    await a.close()
    await idx.close()


async def test_hash_chain_grows_and_verifies(audit):
    sid = "ses_a"
    await audit.open_session(sid)
    await audit.record("gate_decision", session_id=sid, payload={"action": "pass"})
    await audit.record("module_request", session_id=sid, payload={"module": "memory"})
    await audit.record("module_response", session_id=sid, payload={"ok": True})
    await audit.verify_session(sid)


async def test_redaction_applied(audit, tmp_path):
    sid = "ses_b"
    await audit.open_session(sid)
    await audit.record(
        "llm_request", session_id=sid,
        payload={"api_key": "SECRET123", "content": "hi"},
    )
    # read raw jsonl
    files = sorted((tmp_path / "audit" / "sessions" / sid).glob("events.*.jsonl"))
    body = files[-1].read_text()
    assert "SECRET123" not in body
    assert "***" in body


async def test_tamper_detected(audit, tmp_path):
    sid = "ses_c"
    await audit.open_session(sid)
    await audit.record("gate_decision", session_id=sid, payload={"action": "pass"})
    await audit.record("gate_decision", session_id=sid, payload={"action": "pass"})
    # tamper: rewrite one line
    f = sorted((tmp_path / "audit" / "sessions" / sid).glob("events.*.jsonl"))[-1]
    lines = f.read_text().splitlines()
    lines[1] = lines[1].replace("\"pass\"", "\"stop\"")
    f.write_text("\n".join(lines) + "\n")
    with pytest.raises(AuditIntegrityError):
        await audit.verify_session(sid)
