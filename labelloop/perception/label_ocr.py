"""Fallback label read: the printed 5-digit ZIP on the white label, by Tesseract OCR, for QR codes too small for the head camera
(real Miro label: QR ~45 px, ~1.5 px per module, undecodable; its printed ZIP is large). The harness accepts a ZIP only when the
same value is read in consecutive frames (the code Confirmer), since OCR can misread a digit."""
import re
import shutil
import subprocess

import cv2
import numpy as np

TESSERACT = shutil.which("tesseract")


def zips(img_rgb, roi, max_labels=3):
    """[(zip5, (cx, cy))] for the white labels inside `roi` (x0, y0, x1, y1 image fractions); [] without tesseract."""
    if not TESSERACT:
        return []
    h, w = img_rgb.shape[:2]
    x0, y0, x1, y1 = int(roi[0] * w), int(roi[1] * h), int(roi[2] * w), int(roi[3] * h)
    hsv = cv2.cvtColor(img_rgb[y0:y1, x0:x1], cv2.COLOR_RGB2HSV)
    mask = cv2.morphologyEx(((hsv[..., 1] < 50) & (hsv[..., 2] > 170)).astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(mask)
    out = []
    for i in np.argsort(-st[1:, 4])[:max_labels] + 1:
        x, y, bw, bh, area = st[i]
        if area < 800 or bw > 0.4 * w:  # too small to hold readable digits / a white bin wall
            continue
        crop = cv2.resize(cv2.cvtColor(img_rgb[y0 + y:y0 + y + bh, x0 + x:x0 + x + bw], cv2.COLOR_RGB2GRAY), None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        try:
            r = subprocess.run([TESSERACT, "stdin", "stdout", "--psm", "11", "-c", "tessedit_char_whitelist=0123456789"],
                               input=cv2.imencode(".png", crop)[1].tobytes(), capture_output=True, timeout=3)
        except subprocess.TimeoutExpired:
            continue
        out += [(z, (x0 + x + bw / 2, y0 + y + bh / 2)) for z in re.findall(r"(?<!\d)\d{5}(?!\d)", r.stdout.decode())]
    return out
