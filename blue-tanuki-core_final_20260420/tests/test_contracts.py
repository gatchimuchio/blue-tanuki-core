from blue_tanuki_core.contracts import (
    CallContext, ControlRequest, Message, ModuleRequest, TypedPayload,
    new_call_id, new_request_id, new_session_id, new_turn_id,
    payload_hash,
)


def test_id_prefixes():
    assert new_session_id().startswith("ses_")
    assert new_request_id().startswith("req_")
    assert new_turn_id().startswith("turn_")
    assert new_call_id().startswith("call_")


def test_control_request_serialization_roundtrip():
    req = ControlRequest(
        session_id="ses_xxx",
        message=Message(role="user", content="hello"),
    )
    data = req.model_dump_json()
    back = ControlRequest.model_validate_json(data)
    assert back.message.content == "hello"
    assert back.session_id == "ses_xxx"


def test_module_request_requires_context():
    mreq = ModuleRequest(
        request_id="req_1",
        session_id="ses_1",
        turn_id="turn_1",
        module="memory",
        payload=TypedPayload(kind="memory.set", data={"k": "v"}),
        context=CallContext(op="set", resource="k"),
        caller="director",
    )
    assert mreq.context.op == "set"


def test_payload_hash_is_deterministic():
    p1 = {"a": 1, "b": [1, 2]}
    p2 = {"b": [1, 2], "a": 1}
    assert payload_hash(p1) == payload_hash(p2)
