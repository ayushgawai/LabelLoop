import pytest

from labelloop.wallx_client import (WallxClient, WallxError, build_payload, decode_jpeg_b64, encode_jpeg_b64,
                                    hold_response)
from wallx_stub import Stub, rgb

STATE = {"left_ee": {"position": [.1, .2, .3], "rotation_rpy": [0, 0, 0], "gripper": .1},
         "right_ee": {"position": [.1, -.2, .3], "rotation_rpy": [0, 0, 0], "gripper": .1}}


def payload(emb="quanta_x1_pro/gripper-g"):
    return build_payload(emb, "pick", {k: encode_jpeg_b64(rgb()) for k in ("front", "left_wrist", "right_wrist")}, STATE)


def test_handshake_infer_and_error():
    stub = Stub()
    c = WallxClient(stub.url)
    c.connect()
    assert c.metadata["model"] == "wall-x"
    r = c.infer(payload())
    assert len(r["actions"]["left_ee"]["position"]) == 32 and r["actions"]["right_ee"]["gripper"][0] == .1
    with pytest.raises(WallxError) as e:
        c.infer(payload("nope"))
    assert e.value.code == "bad_embodiment"
    assert c.infer(payload())  # connection survives error replies
    c.close()
    stub.stop()


def test_reconnect_after_1011():
    stub = Stub()
    c = WallxClient(stub.url)
    c.connect()
    stub.close_once = True
    assert c.infer(payload())["embodiment"]
    assert stub.connections == 2
    c.close()
    stub.stop()


def test_jpeg_roundtrip_and_prefix():
    s = encode_jpeg_b64(rgb())
    assert decode_jpeg_b64("data:image/jpeg;base64," + s).shape == (48, 64, 3)
    with pytest.raises(ValueError):
        decode_jpeg_b64("bm90IGEganBlZw==")


def test_hold_response_shape():
    a = hold_response("e", STATE)["actions"]
    assert len(a["left_ee"]["position"]) == len(a["right_ee"]["gripper"]) == 32


def test_session_closed_not_resent_and_reconnects():
    stub = Stub()
    stub.close_on_bad = True
    c = WallxClient(stub.url)
    c.connect()
    with pytest.raises(WallxError) as e:
        c.infer(payload("nope"))
    assert e.value.code == "session_closed" and e.value.message == "wallx_session_closed"
    assert len(stub.requests) == 1 and stub.connections == 2  # not resent; reconnected
    assert c.infer(payload())["embodiment"]
    c.close()
    stub.stop()


def test_reset_history_only_when_true():
    assert "reset_history" not in payload() and build_payload("e", "i", {}, STATE, True)["reset_history"] is True
