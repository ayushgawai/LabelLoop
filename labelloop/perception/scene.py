import cv2
import numpy as np


def _mask(img, ranges, k):
    """Union of HSV [lo, hi] ranges (more than one for red, which wraps at H 0), opened with a k x k kernel."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    m = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in ranges:
        m |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))


def inside(p, box):
    return box is not None and box[0] <= p[0] <= box[2] and box[1] <= p[1] <= box[3]


def _frac_box(img, b):
    h, w = img.shape[:2]
    return int(b[0] * w), int(b[1] * h), int(b[2] * w), int(b[3] * h)


def in_bin(p, img, cfg):
    """True if image point p lies in any bin / tray ROI grown by scene.bin_margin (fraction of the image, generous: it includes the walls)."""
    m = cfg.get("bin_margin", 0.03)
    return any(inside(p, _frac_box(img, (b[0] - m, b[1] - m, b[2] + m, b[3] + m))) for b in cfg["bin_rois"].values())


def table_region(img, cfg):
    """(bbox x0, y0, x1, y1, mask) of the detection region. whole_table: the projected tabletop (scene.table, root box [x0, x1, y0, y1]) clipped to the
    image minus every bin / tray ROI (+ bin_margin); else the induction ROI box, all True."""
    h, w = img.shape[:2]
    if not cfg.get("whole_table"):
        x0, y0, x1, y1 = _frac_box(img, cfg["induction_roi"])
        return x0, y0, x1, y1, np.ones((y1 - y0, x1 - x0), bool)
    from labelloop import look  # (late: look needs no scene)
    x0, x1, y0, y1 = cfg["table"]
    poly = np.array([look.table_to_pixel((x, y, look.TABLE_Z), (w, h)) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))], np.float64)
    m = cv2.fillPoly(np.zeros((h, w), np.uint8), [np.round(poly).astype(np.int32)], 255)
    g = cfg.get("bin_margin", 0.03)
    for b in cfg["bin_rois"].values():
        bx0, by0, bx1, by1 = _frac_box(img, (b[0] - g, b[1] - g, b[2] + g, b[3] + g))
        m[max(by0, 0):by1, max(bx0, 0):bx1] = 0
    ys, xs = np.nonzero(m)
    bx0, by0, bx1, by1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    return int(bx0), int(by0), int(bx1), int(by1), m[by0:by1, bx0:bx1] > 0


def roi_box(img, cfg, key="induction_roi", sub=None):
    """Pixel box of cfg[key] (or cfg[key][sub], e.g. a bin colour in bin_rois); the induction ROI is the table region's bbox when whole_table."""
    if key == "induction_roi" and sub is None:
        return table_region(img, cfg)[:4]
    return _frac_box(img, cfg[key] if sub is None else cfg[key][sub])


def roi_area(img, cfg, key="induction_roi", sub=None):
    """Pixels of the ROI the presence / big fractions are normalised by: always the induction ROI box (so the thresholds mean the same with whole_table)."""
    x0, y0, x1, y1 = _frac_box(img, cfg[key] if sub is None else cfg[key][sub])
    return max((x1 - x0) * (y1 - y0), 1)


def zone_ref(img, cfg, key="induction_roi", sub=None):
    """Blurred COLOUR crop of the induction zone (or another ROI): the reference, or the current frame to compare to it.
    Colour, not gray: a kraft box on the green work mat has almost the same gray level as the mat (sim v7 probe: presence 0.0)."""
    x0, y0, x1, y1 = roi_box(img, cfg, key, sub)
    return cv2.GaussianBlur(img[y0:y1, x0:x1], (img.shape[1] // 100 | 1,) * 2, 0)


def presence(img, ref, cfg, key="induction_roi", sub=None):
    """Fraction of the ROI (default the induction zone) that differs from its reference."""
    return float(_changed(img, ref, cfg, key, sub).sum() / roi_area(img, cfg, key, sub))


def _changed(img, ref, cfg, key="induction_roi", sub=None):
    cur = zone_ref(img, cfg, key, sub)
    d = cv2.absdiff(cur, cv2.resize(ref, cur.shape[1::-1])).max(axis=2) > cfg["presence_thr"]  # any channel
    return d & table_region(img, cfg)[4] if key == "induction_roi" and sub is None else d


def centroid(img, ref, cfg):
    """(x, y) of the changed pixels of the induction zone in full-image coords (zone centre if nothing changed)."""
    x0, y0, x1, y1 = roi_box(img, cfg)
    ys, xs = np.nonzero(_changed(img, ref, cfg))
    return (x0 + xs.mean(), y0 + ys.mean()) if len(xs) else ((x0 + x1) / 2, (y0 + y1) / 2)


def changed_box(img, ref, cfg):
    """(x0, y0, x1, y1) of the changed pixels of the induction zone in full-image coords, None if nothing changed."""
    x0, y0, _, _ = roi_box(img, cfg)
    ys, xs = np.nonzero(_changed(img, ref, cfg))
    return (x0 + xs.min(), y0 + ys.min(), x0 + xs.max(), y0 + ys.max()) if len(xs) else None


def changed_mask(img, ref, cfg):
    """Full-image uint8 mask (255) of the changed pixels of the induction zone."""
    x0, y0, x1, y1 = roi_box(img, cfg)
    m = np.zeros(img.shape[:2], np.uint8)
    m[y0:y1, x0:x1] = _changed(img, ref, cfg) * 255
    return m


def kraft_in_roi(img, cfg):
    """Fraction of the induction zone that is kraft-coloured (parcel vs garbage)."""
    x0, y0, x1, y1 = roi_box(img, cfg)
    return float(((_mask(img, [cfg["kraft"]], max(2, img.shape[1] // 200))[y0:y1, x0:x1] > 0) & table_region(img, cfg)[4]).sum() / roi_area(img, cfg))  # logged only: no routing decision uses it
