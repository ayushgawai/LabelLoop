import cv2
import numpy as np


TABLE_GREEN = (46, 120, 100)
MAGENTA = (255, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)


def _clip_box(box, w, h):
    if box is None:
        return None
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0 = max(0, min(w - 1, x0))
    y0 = max(0, min(h - 1, y0))
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    if x1 < x0 or y1 < y0:
        return None
    return x0, y0, x1, y1


def _pad_box(box, pad, w, h):
    if box is None:
        return None
    x0, y0, x1, y1 = box
    return _clip_box((x0 - pad, y0 - pad, x1 + pad, y1 + pad), w, h)


def _center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _draw_rect(img, box, color, thickness):
    if box is None:
        return
    x0, y0, x1, y1 = box
    cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness)


def draw(img_rgb, mode, parcel=None, target=None, bins=None, roi=None, cells=(8, 5)) -> np.ndarray:
    if mode == "none":
        return img_rgb
    out = img_rgb.copy()
    h, w = out.shape[:2]
    parcel = _clip_box(parcel, w, h)
    target = _clip_box(target, w, h)
    bins = bins or {}

    if mode == "mask":
        for name, box in bins.items():
            b = _clip_box(box, w, h)
            if b is not None and (target is None or tuple(b) != tuple(target)) and name != "target":
                x0, y0, x1, y1 = b
                out[y0:y1 + 1, x0:x1 + 1] = TABLE_GREEN

    if parcel is not None:
        _draw_rect(out, _pad_box(parcel, 12, w, h), MAGENTA, 6)
    if target is not None:
        _draw_rect(out, target, GREEN, 6)
    if parcel is not None and target is not None:
        p = tuple(map(int, map(round, _center(parcel))))
        t = tuple(map(int, map(round, _center(target))))
        cv2.arrowedLine(out, p, t, YELLOW, 5, tipLength=0.2)

    if mode == "grid" and roi is not None:
        r = _clip_box(roi, w, h)
        if r is not None:
            x0, y0, x1, y1 = r
            cols, rows = cells
            cols, rows = max(int(cols), 1), max(int(rows), 1)
            for c in range(1, cols):
                x = int(round(x0 + (x1 - x0) * c / cols))
                cv2.line(out, (x, y0), (x, y1), WHITE, 1)
            for r_i in range(1, rows):
                y = int(round(y0 + (y1 - y0) * r_i / rows))
                cv2.line(out, (x0, y), (x1, y), WHITE, 1)
            n = 1
            for r_i in range(rows):
                for c in range(cols):
                    cx0 = int(round(x0 + (x1 - x0) * c / cols))
                    cy0 = int(round(y0 + (y1 - y0) * r_i / rows))
                    cx1 = int(round(x0 + (x1 - x0) * (c + 1) / cols))
                    cy1 = int(round(y0 + (y1 - y0) * (r_i + 1) / rows))
                    cv2.putText(out, str(n), (cx0 + 4, min(cy1 - 4, cy0 + 16)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)
                    n += 1
    return out
