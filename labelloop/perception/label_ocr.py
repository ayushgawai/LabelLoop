"""Fallback label read: the printed 5-digit ZIP on the white label, by Tesseract OCR, for QR codes too small for the head camera
(real Miro label: QR ~45 px, ~1.5 px per module, undecodable; its printed ZIP is large). The harness accepts a ZIP only when the
same value is read in consecutive frames (the code Confirmer), since OCR can misread a digit."""
import re
import shutil
import subprocess

import cv2
import numpy as np

TESSERACT = shutil.which("tesseract")
ZIP = re.compile(r"(?<!\d)\d{5}(?!\d)")


def _ocr(crop, tsv=False):
    pixels = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    cmd = [TESSERACT, "stdin", "stdout", "--psm", "11", "-c", "tessedit_char_whitelist=0123456789"]
    if tsv:
        cmd.append("tsv")
    try:
        result = subprocess.run(cmd, input=cv2.imencode(".png", pixels)[1].tobytes(), capture_output=True, timeout=3)
    except subprocess.TimeoutExpired:
        return ""
    return result.stdout.decode()


def _sideways_zips(crop):
    hits = set()
    for direction in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
        rotated = cv2.rotate(crop, direction)
        height, width = rotated.shape
        # The large ZIP is on the right of these shipping labels. Exclude most of the QR and the smaller print.
        right = rotated[:, width // 2:]
        for region in (right[int(.37 * height):int(.66 * height)], right):
            words = _ocr(region, tsv=True).splitlines()[1:]
            found = {fields[-1] for word in words if len(fields := word.split("\t")) == 12
                     and ZIP.fullmatch(fields[-1]) and float(fields[-2]) >= 65}
            if found:
                hits.update(found)
                break
    return hits if len(hits) == 1 else set()


def zips(img_rgb, roi, max_labels=3):
    """[(zip5, (cx, cy))] for the white labels inside `roi` (x0, y0, x1, y1 image fractions); [] without tesseract."""
    if not TESSERACT:
        return []
    h, w = img_rgb.shape[:2]
    x0, y0, x1, y1 = int(roi[0] * w), int(roi[1] * h), int(roi[2] * w), int(roi[3] * h)
    hsv = cv2.cvtColor(img_rgb[y0:y1, x0:x1], cv2.COLOR_RGB2HSV)
    mask = cv2.morphologyEx(((hsv[..., 1] < 50) & (hsv[..., 2] > 170)).astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(mask)
    out, candidates = [], []
    grey = cv2.cvtColor(img_rgb[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
    for i in np.argsort(-st[1:, 4])[:max_labels] + 1:
        x, y, bw, bh, area = st[i]
        if area < 800 or bw > 0.4 * w:  # too small to hold readable digits / a white bin wall
            continue
        center = (x0 + x + bw / 2, y0 + y + bh / 2)
        out.extend((z, center) for z in ZIP.findall(_ocr(grey[y:y + bh, x:x + bw])))
        pad = max(2, round(6 * w / 1280))
        candidates.append((grey[max(0, y - pad):y + bh + pad, max(0, x - pad):x + bw + pad], center))
    if out:
        return out
    return [(z, center) for crop, center in candidates for z in _sideways_zips(crop)]


def known_zips(csv_path=None):
    """5-digit pincodes printed on our labels (docs/labels/labels.csv): OCR votes prefer these over misreads like 90297 for 90291."""
    import csv
    from pathlib import Path
    p = Path(csv_path) if csv_path else Path(__file__).resolve().parents[2] / "docs/labels/labels.csv"
    try:
        return {r["zip5"] for r in csv.DictReader(open(p, encoding="utf-8")) if r.get("zip5")}
    except OSError:
        return set()


def zip_vote(img_rgb, roi, known=None, max_labels=4):
    """Robust pincode read for small labels: every white label blob in `roi`, several crops (whole label, right half, its middle band)
    x scales (4, 6, 8) x page modes (6, 7, 11); 5-digit reads are counted and the most frequent KNOWN pincode wins (else the most
    frequent read seen at least 3 times). Returns (zip5, votes, center) or None. ~1-2 s."""
    if not TESSERACT:
        return None
    from collections import Counter
    known = known_zips() if known is None else known
    h, w = img_rgb.shape[:2]
    x0, y0, x1, y1 = int(roi[0] * w), int(roi[1] * h), int(roi[2] * w), int(roi[3] * h)
    hsv = cv2.cvtColor(img_rgb[y0:y1, x0:x1], cv2.COLOR_RGB2HSV)
    mask = cv2.morphologyEx(((hsv[..., 1] < 50) & (hsv[..., 2] > 170)).astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(mask)
    grey = cv2.cvtColor(img_rgb[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
    votes, where = Counter(), {}
    blobs = [i for i in range(1, n) if st[i][4] >= 800 and 0.4 < st[i][2] / max(1, st[i][3]) < 3 and st[i][4] > 0.45 * st[i][2] * st[i][3]]
    for i in sorted(blobs, key=lambda i: -st[i][4])[:max_labels]:  # label-shaped and filled: not the long thin zone tape
        x, y, bw, bh, _ = st[i]
        lab = grey[y:y + bh, x:x + bw]
        for crop in (lab, lab[:, bw * 2 // 5:], lab[bh // 3:bh * 5 // 6, bw * 2 // 5:]):
            for f in (4, 6, 8):
                big = cv2.resize(crop, None, fx=f / 4, fy=f / 4, interpolation=cv2.INTER_CUBIC)  # _ocr upsamples x4 again
                for psm in ("6", "7", "11"):
                    try:
                        r = subprocess.run([TESSERACT, "stdin", "stdout", "--psm", psm, "-c", "tessedit_char_whitelist=0123456789"],
                                           input=cv2.imencode(".png", cv2.resize(big, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC))[1].tobytes(),
                                           capture_output=True, timeout=3)
                    except subprocess.TimeoutExpired:
                        continue
                    for z in ZIP.findall(r.stdout.decode()):
                        votes[z] += 1
                        where[z] = (x0 + x + bw / 2, y0 + y + bh / 2)
    good = [(c, z) for z, c in votes.items() if z in known] or [(c, z) for z, c in votes.items() if c >= 3]
    if not good:
        return None
    c, z = max(good)
    return z, c, where[z]
