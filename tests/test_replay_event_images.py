import sys
from pathlib import Path

import cv2
import numpy as np
import qrcode

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import replay_event_images as r


def qr_img(text, n=140):
    return np.array(qrcode.make(text).convert("RGB").resize((n, n)))


def test_bin_codes_ignored_table_codes_accepted(tmp_path):
    ep = tmp_path / "ep1"
    ep.mkdir()
    empty = np.full((480, 640, 3), 90, np.uint8)
    cur = empty.copy()
    cur[10:150, 130:270] = qr_img("BINCODE")   # centre (200, 80): inside blue bin ROI
    cur[300:440, 380:520] = qr_img("TABLECODE")  # centre (450, 370): on the table
    for name, im in (("0000", empty), ("0001", cur)):
        cv2.imwrite(str(ep / f"{name}_front.jpg"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(ep / "0001_left_wrist.jpg"), cv2.cvtColor(qr_img("WRIST", 200), cv2.COLOR_RGB2BGR))
    rows = r.run(tmp_path)
    res = dict((c, x) for _, c, x in rows)["0001"]
    assert [t for t, _ in res["accepted"]] == ["TABLECODE"]
    assert [t for t, _ in res["ignored"]] == ["BINCODE"]
    assert res["frac"] > 0 and (ep / "annotated" / "0001_front.png").exists()
