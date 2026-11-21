import cv2
import zxingcpp


def locate(img, upscale=True):
    """[(text, (cx, cy))] for every QR in one RGB image. zxing first; OpenCV only if zxing finds nothing; then an upscaled, sharpened retry."""
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
    if not out and upscale:  # real head camera: a label is ~60 px, oblique, taped (glare); x3 + unsharp reads it (trial 2: every unoccluded frame)
        s = min(3.0, 3840 / max(img.shape[:2]))
        g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        g = cv2.addWeighted(g, 1.8, cv2.GaussianBlur(g, (0, 0), 2), -0.8, 0)
        out = [(b.text, (sum(p.x for p in (b.position.top_left, b.position.bottom_right)) / 2 / s,
                         sum(p.y for p in (b.position.top_left, b.position.bottom_right)) / 2 / s))
               for b in zxingcpp.read_barcodes(g, formats=zxingcpp.BarcodeFormat.QRCode)]
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
