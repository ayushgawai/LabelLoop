"""Generate QR labels, box textures and a printable A4 sheet. Self-checks every texture by decoding it.

Cube CUBE_MM; label = QR symbol (QR_MM wide) + MARGIN_MM white quiet zone on every side (QR_MM + 2*MARGIN_MM =
50 mm), centred on the top face and on the +Y side face.
"""
import argparse
import math
from pathlib import Path

import numpy as np
import qrcode
import zxingcpp
from PIL import Image, ImageDraw, ImageFont

CUBE_MM = 60.0  # parcel edge; G gripper opens ~97 mm
QR_MM, MARGIN_MM = 40.0, 5.0
FACE_MM = SIDE_MM = (CUBE_MM, CUBE_MM)
A4_MM = (210, 297)
KRAFT = [(181, 142, 96), (170, 132, 88), (190, 152, 106)]  # [0] boxes, [1:] decoys


def code_for(zone, n):
    return f"LL|ZONE-{zone}|{n:03d}"


def safe(code):
    return code.replace("|", "_")


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
    ap.add_argument("--per-zone", type=int, default=3)
    ap.add_argument("--dpi", type=int, default=300)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    codes = [code_for(z, n) for z in "AB" for n in range(1, a.per_zone + 1)]
    for code in codes:
        s = safe(code)
        lab = label_image(code, QR_MM, a.dpi)
        lab.save(out / f"{s}_qr.png", dpi=(a.dpi, a.dpi))
        face = texture(FACE_MM, a.dpi, KRAFT[0], lab)
        side = texture(SIDE_MM, a.dpi, KRAFT[0], lab)
        for name, img in (("qr", lab), ("face", face), ("side", side)):
            check(img, [code], f"{s}_{name}")
        face.save(out / f"{s}_face.png", dpi=(a.dpi, a.dpi))
        side.save(out / f"{s}_side.png", dpi=(a.dpi, a.dpi))
    for i, color in enumerate(KRAFT[1:], 1):
        img = texture(FACE_MM, a.dpi, color)
        check(img, [], f"decoy_{i}")
        img.save(out / f"decoy_{i}.png", dpi=(a.dpi, a.dpi))
    page = sheet(codes, 300)
    got = sorted(decode(page))
    if got != sorted(codes):
        raise SystemExit(f"FAIL sheet: decoded {got}")
    page.save(out / "sheet_a4.png", dpi=(300, 300))
    print(f"wrote {len(codes)} codes + 2 decoys + sheet_a4.png to {out}; all decode round-trips OK")


if __name__ == "__main__":
    main()
