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
        assert read(tmp_path / f"{m.safe(code)}_label.png") == [code]
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
    assert m.QR_MM == 40 and m.CUBE_MM == 60
    assert all(read(tmp_path / f"{m.safe(r['qr_payload'])}_label.png") == [r["qr_payload"]] for r in m.LABELS)  # spares too


def test_label_sheet_pdf_pages_decode_every_payload(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
    import make_labels as ml
    ml.main(["--out", str(tmp_path)])
    pg = ml.pages()
    assert all(p.size == (2550, 3300) for p in pg) and len(pg) <= 3
    assert sorted(t for p in pg for t in m.decode(p)) == sorted(r["qr_payload"] for r in m.LABELS)
    assert (tmp_path / "labels_letter.pdf").read_bytes()[:4] == b"%PDF" and len((tmp_path / "labels.csv").read_text().splitlines()) == len(m.LABELS) + 1
    # large label QR: quiet zone >= 4 modules, 40 mm symbol
    assert Image.open(tmp_path / "preview_large.png").size == (1181, 827) and Image.open(tmp_path / "preview_small.png").size == (732, 732)
