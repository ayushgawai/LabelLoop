import sys
from pathlib import Path

import numpy as np
import zxingcpp
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
import make_qr_boxes as m  # noqa: E402


def read(path):
    return [r.text for r in zxingcpp.read_barcodes(np.array(Image.open(path).convert("L")))]


def test_generate_and_roundtrip(tmp_path):
    m.main(["--out", str(tmp_path)])
    for name, code in m.ROUTES.items():  # lane-sort payloads 420/<ZIP5>/<seq3>
        assert read(tmp_path / f"{m.safe(code)}_qr.png") == [code]
        X, Y, Z = (v * 1000 for v in m.PARCELS[name]["size"])
        mm = {"pz": (X, Y), "nz": (X, Y), "py": (X, Z), "ny": (X, Z), "px": (Y, Z), "nx": (Y, Z)}
        for vid, p in m.variants(name).items():  # six face tiles at real size; the code decodes on exactly the variant's face
            assert p["labels"] == (vid.rsplit("_", 1)[1],)
            for f in m.FACES:
                tile = tmp_path / f"box_{vid}_{f}.png"
                assert Image.open(tile).size == tuple(round(v * m.PPM) for v in mm[f])
                assert read(tile) == ([code] if f in p["labels"] else [])
    for name in m.PARCELS.keys() - m.ROUTES.keys():  # unlabelled props: no code on any face
        assert all(read(tmp_path / f"box_{name}_{f}.png") == [] for f in m.FACES)
    assert read(tmp_path / "nolabel.png") == []
    assert round(m.QR_MM + 2 * m.MARGIN_MM) == 50 and m.CUBE_MM == 60
    assert Image.open(tmp_path / "sheet_a4.png").size == (2480, 3508)
    assert sorted(read(tmp_path / "sheet_a4.png")) == sorted(m.ROUTES.values())  # printable sheet = the parcel labels
