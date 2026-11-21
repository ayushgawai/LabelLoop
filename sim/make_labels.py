"""Print sheet + mapping for the shipping labels (props.LABELS): docs/labels/labels.csv, labels_letter.pdf (US Letter, 300 dpi, true size),
preview_large.png / preview_small.png. Self-checks: every payload decodes from the rendered pages.
Usage: python sim/make_labels.py [--out docs/labels]
"""
import argparse
import csv
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_qr_boxes as mq  # noqa: E402
from props import LABEL_SIZES, LABELS  # noqa: E402

DPI = 300
LETTER_MM = (215.9, 279.4)
MARGIN, GUTTER, FOOT = 8.0, 6.0, 18.0  # page margin, gap between labels, footer band (mm)
COLS = ["label_id", "qr_payload", "zip5", "zip3", "lane", "target_bin", "recipient", "street", "city", "state", "tracking", "weight_lb", "date",
        "size", "sim_id", "item"]


def pages(dpi=DPI):
    """Letter pages (PIL RGB). Left column: large labels stacked; right column (70 mm wide): small labels, then large ones turned 90 deg."""
    k = dpi / 25.4
    big = [r for r in LABELS if r["size"] == "large"]
    small = [r for r in LABELS if r["size"] == "small"]
    lw, lh = LABEL_SIZES["large"]
    bottom = LETTER_MM[1] - FOOT
    out = []
    while big or small:
        page = Image.new("RGB", (round(LETTER_MM[0] * k), round(LETTER_MM[1] * k)), "white")
        d = ImageDraw.Draw(page)

        def put(r, x, y, turn=False):
            img = mq.label_image(r, dpi)
            img = img.rotate(90, expand=True) if turn else img
            X, Y = round(x * k), round(y * k)
            page.paste(img, (X, Y))
            d.rectangle([X - 2, Y - 2, X + img.width + 1, Y + img.height + 1], outline=(170, 170, 170))  # cut line
            return img.height / k

        y = MARGIN
        while big and y + lh <= bottom:
            put(big.pop(0), MARGIN, y)
            y += lh + GUTTER
        x, y = MARGIN + lw + GUTTER, MARGIN
        while small and y + LABEL_SIZES["small"][1] <= bottom:
            y += put(small.pop(0), x, y) + GUTTER
        while big and y + lw <= bottom:
            y += put(big.pop(0), x, y, turn=True) + GUTTER
        fy = bottom + 6  # footer: 50 mm ruler + text
        f = ImageFont.load_default(size=round(3.2 * k))
        d.line([MARGIN * k, fy * k, (MARGIN + 50) * k, fy * k], fill="black", width=3)
        for t in range(0, 51, 10):
            d.line([(MARGIN + t) * k, (fy - 2) * k, (MARGIN + t) * k, (fy + 2) * k], fill="black", width=3)
        d.text((MARGIN * k, (fy + 2.5) * k), "50 mm", fill="black", font=f)
        d.text(((MARGIN + 60) * k, fy * k), f"Print at 100% / Actual size - check the 50 mm ruler.   LabelLoop labels, page {len(out) + 1}",
               fill="black", font=f, anchor="lm")
        out.append(page)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/labels")
    out = Path(ap.parse_args(argv).out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "labels.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, COLS)
        w.writeheader()
        w.writerows({c: r[c] for c in COLS} for r in LABELS)
    pg = pages()
    got = sorted(t for p in pg for t in mq.decode(p))
    if got != sorted(r["qr_payload"] for r in LABELS):
        raise SystemExit(f"FAIL pages: decoded {got}")
    pg[0].save(out / "labels_letter.pdf", save_all=True, append_images=pg[1:], resolution=DPI)
    for size in LABEL_SIZES:
        mq.label_image(next(r for r in LABELS if r["size"] == size), DPI).save(out / f"preview_{size}.png", dpi=(DPI, DPI))
    print(f"wrote {len(LABELS)} labels on {len(pg)} pages to {out}; all payloads decode")


if __name__ == "__main__":
    main()
