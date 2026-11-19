import cv2
import zxingcpp


def decode(images):
    """(text, view) for every QR found. zxing first; OpenCV only if zxing finds nothing in that image."""
    found = []
    for view, img in images.items():
        texts = [b.text for b in zxingcpp.read_barcodes(img, formats=zxingcpp.BarcodeFormat.QRCode)]
        if not texts:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            ok, info, _, _ = cv2.QRCodeDetector().detectAndDecodeMulti(gray)
            texts = [t for t in info if t] if ok else []
        found += [(t, view) for t in texts]
    return found


class Confirmer:
    """Returns codes seen in n consecutive updates."""

    def __init__(self, n=2):
        self.n, self.counts = n, {}

    def update(self, codes):
        self.counts = {c: self.counts.get(c, 0) + 1 for c in codes}
        return {c for c, k in self.counts.items() if k >= self.n}
