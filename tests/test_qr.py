import numpy as np
import qrcode
import zxingcpp

from labelloop.perception import qr


def img(code, size=200):
    q = np.asarray(qrcode.make(code).convert("RGB").resize((size, size)))
    out = np.full((480, 640, 3), 128, np.uint8)
    out[100:100 + size, 100:100 + size] = q
    return out


def test_decode_finds_code_and_view():
    blank = np.full((480, 640, 3), 128, np.uint8)
    assert qr.decode({"front": blank, "right_wrist": img("LL|ZONE-A|001")}) == [("LL|ZONE-A|001", "right_wrist")]


def test_opencv_fallback_when_zxing_finds_nothing(monkeypatch):
    monkeypatch.setattr(zxingcpp, "read_barcodes", lambda *a, **k: [])
    assert qr.decode({"front": img("LL|ZONE-B|002")}) == [("LL|ZONE-B|002", "front")]


def test_nothing_found():
    assert qr.decode({"front": np.full((480, 640, 3), 128, np.uint8)}) == []


def test_confirmer_needs_consecutive_frames():
    c = qr.Confirmer(2)
    assert c.update({"a"}) == set()
    assert c.update({"a", "b"}) == {"a"}
    assert c.update({"b"}) == {"b"}
    assert c.update({"a", "b"}) == {"b"}   # a's streak was reset when it dropped out
