"""Generate the lane-sort parcel QR labels, box textures and a printable A4 label sheet. Self-checks every image by decoding it.

Label = QR symbol (QR_MM wide) + MARGIN_MM white quiet zone on every side (QR_MM + 2*MARGIN_MM = 50 mm). Payload 420/<ZIP5>/<seq3>
(GS1 AI 420 = ship-to ZIP). A parcel without a measured prop is a CUBE_MM cube: face + side texture, label centred on both.
A measured parcel (qr_sort_task/props.py PARCELS) gets six face tiles box_<id>_<face>.png at PPM px/mm: its body colour, its tape
stripe, and the label centred on its `labels` face. A labelled parcel has one variant id `<id>_<face>` per face that can hold the label
(props.variants), so the label sits on a random face each episode.
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
from props import FACES, MARGIN_MM, PARCELS, QR_MM, ROUTES, variants  # noqa: E402

CUBE_MM = 60.0  # parcel edge; G gripper opens ~97 mm
FACE_MM = SIDE_MM = (CUBE_MM, CUBE_MM)
A4_MM = (210, 297)
KRAFT = (181, 142, 96)
PPM = 5  # px per mm of measured-parcel face tiles: 40 mm QR -> 200 px, 8 px per module
# tile axes of each face, as make_usd.box_mesh maps them: u = (box axis, +1 | -1 direction), v = box axis (always +; image rows run -v)
TILE_U = {"pz": ("x", 1), "nz": ("x", -1), "py": ("x", -1), "ny": ("x", 1), "px": ("y", 1), "nx": ("y", -1)}
TILE_V = {"pz": "y", "nz": "y", "py": "z", "ny": "z", "px": "z", "nx": "z"}


def safe(code):
    return code.replace("/", "_")


def px(mm, dpi):
    return round(mm * dpi / 25.4)


def label_image(code, symbol_mm, dpi):
    """White square label: QR symbol of symbol_mm with the proportional quiet zone (5 mm at 40 mm)."""
    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_H, border=0)
    qr.add_data(code)
    qr.make(fit=True)
    m = np.array(qr.get_matrix(), dtype=np.uint8)
    sym = px(symbol_mm, dpi)
    img = Image.fromarray(np.where(m, 0, 255).astype(np.uint8)).resize((sym, sym), Image.NEAREST)
    margin = px(MARGIN_MM * symbol_mm / QR_MM, dpi)
    out = Image.new("L", (sym + 2 * margin,) * 2, 255)
    out.paste(img, (margin, margin))
    return out.convert("RGB")


def texture(size_mm, dpi, color, label=None):
    img = Image.new("RGB", (px(size_mm[0], dpi), px(size_mm[1], dpi)), color)
    if label is not None:
        img.paste(label, ((img.width - label.width) // 2, (img.height - label.height) // 2))
    return img


def parcel_faces(p, code):
    """{face: tile} of a measured parcel; tile = the face seen from outside, u/v as TILE_U / TILE_V."""
    mm = dict(zip("xyz", (v * 1000 for v in p["size"])))
    lab = label_image(code, QR_MM, PPM * 25.4) if code else None
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
            img.paste(lab, ((img.width - lab.width) // 2, (img.height - lab.height) // 2))
        out[f] = img
    return out


def decode(img):
    return [r.text for r in zxingcpp.read_barcodes(np.array(img.convert("L")))]


def check(img, expected, name):
    got = decode(img)
    if got != expected:
        raise SystemExit(f"FAIL {name}: decoded {got}, expected {expected}")


def sheet(codes, dpi):
    w, h = px(A4_MM[0], dpi), px(A4_MM[1], dpi)
    page = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(page)
    font = ImageFont.load_default(size=px(4, dpi))
    cols, cell = 2, QR_MM + 2 * MARGIN_MM
    rows = math.ceil(len(codes) / cols)
    if rows * (cell + 6) > A4_MM[1] - 20:
        raise SystemExit(f"{len(codes)} labels do not fit one A4 sheet")
    x0 = (A4_MM[0] - cols * cell) / 2
    for i, code in enumerate(codes):
        x, y = px(x0 + (i % cols) * cell, dpi), px(10 + (i // cols) * (cell + 6), dpi)
        lab = label_image(code, QR_MM, dpi)
        page.paste(lab, (x, y))
        d.rectangle([x, y, x + lab.width - 1, y + lab.height - 1], outline=(160, 160, 160))
        d.text((x + px(2, dpi), y + lab.height + px(0.5, dpi)), code, fill="black", font=font)
    return page


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="sim/assets")
    ap.add_argument("--dpi", type=int, default=300)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    codes = list(ROUTES.values())
    n = 0
    for base in PARCELS:  # measured props: six face tiles per variant (label on its one face)
        for vid, p in variants(base).items():
            for f, img in parcel_faces(p, ROUTES.get(base)).items():
                check(img, [ROUTES[base]] if f in p["labels"] else [], f"box_{vid}_{f}")
                img.save(out / f"box_{vid}_{f}.png")
            n += 1
    img = texture(FACE_MM, a.dpi, KRAFT)  # plain kraft cube
    check(img, [], "nolabel")
    img.save(out / "nolabel.png", dpi=(a.dpi, a.dpi))
    for code in codes:  # the bare label, at print resolution (QR gate, label sheet)
        lab = label_image(code, QR_MM, a.dpi)
        check(lab, [code], f"{safe(code)}_qr")
        lab.save(out / f"{safe(code)}_qr.png", dpi=(a.dpi, a.dpi))
    page = sheet(codes, 300)
    got = sorted(decode(page))
    if got != sorted(codes):
        raise SystemExit(f"FAIL sheet: decoded {got}")
    page.save(out / "sheet_a4.png", dpi=(300, 300))
    print(f"wrote {len(codes)} parcel labels, {n} parcel variants x 6 face tiles, nolabel + sheet_a4.png to {out}; all decode round-trips OK")


if __name__ == "__main__":
    main()
