"""Generate the lane-sort shipping labels, parcel box textures and per-label PNGs. Self-checks every image by decoding it.

Label = fictional "LabelLoop Express" shipping label (props.LABELS registry, sizes props.LABEL_SIZES): QR symbol QR_MM wide with a white
quiet zone of max(MARGIN_MM, 4 modules), plus FROM / SHIP TO / tracking text. Payload 420/<ZIP5>/<seq3> (GS1 AI 420 = ship-to ZIP).
label_image() is the ONE artwork used for the printed sheet (sim/make_labels.py) and the sim textures. A parcel without a measured
prop is a CUBE_MM cube. A measured parcel (qr_sort_task/props.py PARCELS) gets six face tiles box_<id>_<face>.png at PPM px/mm: its
body colour, its tape stripe, and the label centred on its `labels` face (turned 90 deg if only that way round fits). A labelled parcel
has one variant id `<id>_<face>` per face that can hold the label (props.variants), so the label sits on a random face each episode.
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import qrcode
import zxingcpp
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent / "qr_sort_task"))
from props import FACES, LABEL_OF, LABEL_SIZES, LABELS, MARGIN_MM, PARCELS, QR_MM, ROUTES, SENDER, variants  # noqa: E402

CUBE_MM = 60.0  # parcel edge; G gripper opens ~97 mm
FACE_MM = SIDE_MM = (CUBE_MM, CUBE_MM)
KRAFT = (181, 142, 96)
PPM = 5  # px per mm of measured-parcel face tiles: 40 mm QR -> 200 px, 8 px per module
# tile axes of each face, as make_usd.box_mesh maps them: u = (box axis, +1 | -1 direction), v = box axis (always +; image rows run -v)
TILE_U = {"pz": ("x", 1), "nz": ("x", -1), "py": ("x", -1), "ny": ("x", 1), "px": ("y", 1), "nx": ("y", -1)}
TILE_V = {"pz": "y", "nz": "y", "py": "z", "ny": "z", "px": "z", "nx": "z"}


def safe(code):
    return code.replace("/", "_")


def px(mm, dpi):
    return round(mm * dpi / 25.4)


def label_image(r, dpi):
    """Shipping label of registry row r (props.LABELS) as an RGB image at dpi: black on white, thin border."""
    W, H = LABEL_SIZES[r["size"]]
    k = dpi / 25.4
    img = Image.new("RGB", (round(W * k), round(H * k)), "white")
    d = ImageDraw.Draw(img)

    def text(x, y, s, h, w, bold=False, right=False):  # h = font size mm, shrunk until the text fits w mm
        while True:
            f = ImageFont.load_default(size=max(1, round(h * k)))
            sw = round(0.02 * h * k) if bold else 0
            if d.textlength(s, font=f) + 2 * sw <= w * k or h < 1:
                break
            h *= 0.95
        d.text((x * k, y * k), s, fill="black", font=f, anchor="ra" if right else "la", stroke_width=sw, stroke_fill="black")

    def rule(y, x0, x1):
        d.line([x0 * k, y * k, x1 * k, y * k], fill="black", width=max(1, round(0.25 * k)))

    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_H, border=0)
    qr.add_data(r["qr_payload"])
    qr.make(fit=True)
    m = np.array(qr.get_matrix(), dtype=np.uint8)
    q = max(MARGIN_MM, 4 * QR_MM / len(m))  # quiet zone, at least 4 modules
    sym = px(QR_MM, dpi)
    big = r["size"] == "large"
    qx = q if big else (W - QR_MM) / 2
    img.paste(Image.fromarray(np.where(m, 0, 255).astype(np.uint8)).resize((sym, sym), Image.NEAREST).convert("RGB"), (round(qx * k), round(q * k)))
    below = 2 * q + QR_MM  # first y under the QR quiet zone
    if big:
        text(2, below + 0.6, r["tracking"], 2.8, below - 3, bold=True)
        text(2, below + 4.8, f"ID {r['label_id']}   {r['weight_lb']} LB   {r['date']}", 2.1, below - 3)
        x, w = below + 1.5, W - below - 3.5
        text(x, 1.8, "LabelLoop Express", 4.2, w, bold=True)
        text(x, 7.4, "PRIORITY GROUND", 2.4, w)
        rule(11, x, x + w)
        text(x, 11.8, "FROM", 1.8, w)
        for i, line in enumerate(SENDER):
            text(x, 13.8 + 2.5 * i, line, 2.2, w)
        rule(22, x, x + w)
        text(x, 22.8, "SHIP TO", 1.8, w)
        text(x, 25.2, r["recipient"], 3.4, w, bold=True)
        text(x, 29.4, r["street"], 2.7, w)
        text(x, 33, f"{r['city']}, {r['state']}", 2.7, w)
        text(x, 37.5, r["zip5"], 14, w, bold=True)
        rule(55, x, x + w)
        text(x, 56.5, f"WT {r['weight_lb']} LB   SHIP {r['date']}", 2.2, w)
        text(x, 60.5, f"SERVICE PRIORITY GROUND   {r['label_id']}", 2.2, w)
    else:
        text(2.5, below + 0.8, r["recipient"], 2.8, W - 5 - 15, bold=True)
        text(W - 2.5, below + 0.4, r["zip5"], 4.4, 14, bold=True, right=True)
        text(2.5, below + 5.6, r["tracking"], 2.3, W - 5)
    d.rectangle([0, 0, img.width - 1, img.height - 1], outline=(40, 40, 40), width=max(1, round(0.25 * k)))
    return img


def texture(size_mm, dpi, color, label=None):
    img = Image.new("RGB", (px(size_mm[0], dpi), px(size_mm[1], dpi)), color)
    if label is not None:
        img.paste(label, ((img.width - label.width) // 2, (img.height - label.height) // 2))
    return img


def parcel_faces(p, rec):
    """{face: tile} of a measured parcel; tile = the face seen from outside, u/v as TILE_U / TILE_V."""
    mm = dict(zip("xyz", (v * 1000 for v in p["size"])))
    lab = label_image(rec, PPM * 25.4) if rec else None
    t = p.get("tape")
    out = {}
    for f in FACES:
        (ua, us), va = TILE_U[f], TILE_V[f]
        img = Image.new("RGB", (round(mm[ua] * PPM), round(mm[va] * PPM)), tuple(p["color"]))
        if t and t["axis"] in (ua, va):  # the stripe crosses this face
            band = round(t["mm"] * PPM)
            d = ImageDraw.Draw(img)
            if t["axis"] == va:  # horizontal stripe; image row grows as v falls
                y0 = math.floor((img.height - band) / 2 - t["at"] * PPM)
                d.rectangle([0, y0, img.width - 1, y0 + band - 1], fill=tuple(t["rgb"]))
            else:  # vertical stripe
                x0 = math.floor((img.width - band) / 2 + us * t["at"] * PPM)
                d.rectangle([x0, 0, x0 + band - 1, img.height - 1], fill=tuple(t["rgb"]))
        for q in (q for q in p.get("patches", ()) if q["face"] == f):  # printed rectangles, centre (u, v) mm from the face centre
            w, h = (round(v * PPM) for v in q["mm"])
            x0 = math.floor(img.width / 2 + q["at"][0] * PPM - w / 2)
            y0 = math.floor(img.height / 2 - q["at"][1] * PPM - h / 2)
            ImageDraw.Draw(img).rectangle([x0, y0, x0 + w - 1, y0 + h - 1], fill=tuple(q["rgb"]))
        if lab is not None and f in p["labels"]:
            tl = lab if lab.width <= img.width and lab.height <= img.height else lab.rotate(90, expand=True)  # fits only turned
            img.paste(tl, ((img.width - tl.width) // 2, (img.height - tl.height) // 2))
        out[f] = img
    return out


def decode(img):
    return [r.text for r in zxingcpp.read_barcodes(np.array(img.convert("L")))]


def check(img, expected, name):
    got = decode(img)
    if got != expected:
        raise SystemExit(f"FAIL {name}: decoded {got}, expected {expected}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="sim/assets")
    ap.add_argument("--dpi", type=int, default=300)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for base in PARCELS:  # measured props: six face tiles per variant (label on its one face)
        for vid, p in variants(base).items():
            for f, img in parcel_faces(p, LABEL_OF.get(base)).items():
                check(img, [ROUTES[base]] if f in p["labels"] else [], f"box_{vid}_{f}")
                img.save(out / f"box_{vid}_{f}.png")
            n += 1
    img = texture(FACE_MM, a.dpi, KRAFT)  # plain kraft cube
    check(img, [], "nolabel")
    img.save(out / "nolabel.png", dpi=(a.dpi, a.dpi))
    for r in LABELS:  # every registry label at print resolution
        lab = label_image(r, a.dpi)
        check(lab, [r["qr_payload"]], r["label_id"])
        lab.save(out / f"{safe(r['qr_payload'])}_label.png", dpi=(a.dpi, a.dpi))
    print(f"wrote {len(LABELS)} labels, {n} parcel variants x 6 face tiles, nolabel to {out}; all decode round-trips OK")


if __name__ == "__main__":
    main()
