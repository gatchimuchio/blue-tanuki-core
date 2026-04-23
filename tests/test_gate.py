from pathlib import Path

import pytest

from blue_tanuki_core.contracts import GateSubject
from blue_tanuki_core.errors import ConfigError
from blue_tanuki_core.gate import (
    AllowAllGate, DenyAllGate, load_policy, PolicyGate, CompiledRule, RawRule,
)


def _subject(**overrides) -> GateSubject:
    base = dict(
        direction="pre_module",
        session_id="s", request_id="r", turn_id="t",
        caller="director", module="file_ops", op="write",
        resource="/etc/passwd",
        payload_hash="0" * 64,
    )
    base.update(overrides)
    return GateSubject(**base)


def test_allow_all_passes():
    d = AllowAllGate().evaluate(_subject())
    assert d.action == "pass"


def test_deny_all_stops():
    d = DenyAllGate().evaluate(_subject())
    assert d.action == "stop"


def test_policy_loads_example(tmp_path):
    src = Path(__file__).parent.parent / "policy.example.yaml"
    gate = load_policy(src)
    assert isinstance(gate, PolicyGate)


def test_policy_denies_etc_write(tmp_path):
    src = Path(__file__).parent.parent / "policy.example.yaml"
    gate = load_policy(src)
    d = gate.evaluate(_subject(resource="/etc/passwd"))
    assert d.action == "stop"
    assert "deny_system_path_write" in d.matched_rule_ids


def test_policy_passes_safe_write(tmp_path):
    src = Path(__file__).parent.parent / "policy.example.yaml"
    gate = load_policy(src)
    d = gate.evaluate(_subject(resource="/tmp/foo"))
    assert d.action == "pass"


def test_policy_approval_on_web_fetch():
    src = Path(__file__).parent.parent / "policy.example.yaml"
    gate = load_policy(src)
    d = gate.evaluate(_subject(module="web_fetch", op="get", resource="https://x"))
    assert d.action == "suspend_approval"


def test_policy_stop_on_filtered_llm():
    src = Path(__file__).parent.parent / "policy.example.yaml"
    gate = load_policy(src)
    d = gate.evaluate(_subject(direction="post_llm", caller="llm",
                               module=None, op=None, resource=None,
                               finish_reason="filtered"))
    assert d.action == "stop"


def test_severity_stop_beats_approval():
    rules = [
        CompiledRule(RawRule(id="a", when={"direction": "pre_module"},
                             action="suspend_approval")),
        CompiledRule(RawRule(id="b", when={"direction": "pre_module"},
                             action="stop", reason="no")),
    ]
    gate = PolicyGate(rules)
    d = gate.evaluate(_subject())
    assert d.action == "stop"
    assert set(d.matched_rule_ids) == {"a", "b"}


def test_unknown_when_key_raises():
    with pytest.raises(ConfigError):
        CompiledRule(RawRule(id="x", when={"bogus": 1}, action="pass"))


def test_unknown_action_raises():
    with pytest.raises(ConfigError):
        CompiledRule(RawRule(id="x", when={}, action="boom"))
