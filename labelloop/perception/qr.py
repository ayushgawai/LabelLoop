import cv2
import zxingcpp


def locate(img):
    """[(text, (cx, cy))] for every QR in one RGB image. zxing first; OpenCV only if zxing finds nothing."""
    out = []
    for b in zxingcpp.read_barcodes(img, formats=zxingcpp.BarcodeFormat.QRCode):
        p = b.position
        pts = [(p.top_left.x, p.top_left.y), (p.top_right.x, p.top_right.y),
               (p.bottom_right.x, p.bottom_right.y), (p.bottom_left.x, p.bottom_left.y)]
        out.append((b.text, (sum(x for x, _ in pts) / 4, sum(y for _, y in pts) / 4)))
    if not out:
        ok, info, pts, _ = cv2.QRCodeDetector().detectAndDecodeMulti(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
        if ok:
            out = [(t, tuple(float(v) for v in q.mean(axis=0))) for t, q in zip(info, pts) if t]
    return out


def decode(images):
    """(text, view) for every QR found in each image."""
    return [(t, view) for view, img in images.items() for t, _ in locate(img)]


class Confirmer:
    """Returns codes seen in n consecutive updates."""

    def __init__(self, n=2):
        self.n, self.counts = n, {}

    def update(self, codes):
        self.counts = {c: self.counts.get(c, 0) + 1 for c in codes}
        return {c for c, k in self.counts.items() if k >= self.n}
