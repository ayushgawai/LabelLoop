import sys
from pathlib import Path

import numpy as np
import zxingcpp
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
import make_qr_boxes as m  # noqa: E402


def test_generate_and_roundtrip(tmp_path):
    m.main(["--out", str(tmp_path), "--per-zone", "2"])
    for code in ("LL|ZONE-A|001", "LL|ZONE-A|002", "LL|ZONE-B|001", "LL|ZONE-B|002"):
        s = m.safe(code)
        for kind in ("qr", "face", "side"):
            img = Image.open(tmp_path / f"{s}_{kind}.png").convert("L")
            assert [r.text for r in zxingcpp.read_barcodes(np.array(img))] == [code]
    for code in m.ROUTES.values():  # lane-sort payloads 420/<ZIP5>/<seq3>
        for kind in ("qr", "face", "side"):
            img = Image.open(tmp_path / f"{m.safe(code)}_{kind}.png").convert("L")
            assert [r.text for r in zxingcpp.read_barcodes(np.array(img))] == [code]
    assert zxingcpp.read_barcodes(np.array(Image.open(tmp_path / "nolabel.png").convert("L"))) == []
    assert Image.open(tmp_path / "LL_ZONE-A_001_face.png").size == (709, 709)
    assert Image.open(tmp_path / "LL_ZONE-A_001_side.png").size == (709, 709)
    for d in ("decoy_1.png", "decoy_2.png"):
        assert zxingcpp.read_barcodes(np.array(Image.open(tmp_path / d).convert("L"))) == []
    assert round(m.QR_MM + 2 * m.MARGIN_MM) == 50 and m.CUBE_MM == 60
    assert Image.open(tmp_path / "sheet_a4.png").size == (2480, 3508)
