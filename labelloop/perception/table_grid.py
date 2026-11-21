"""Table grid: the green work mat in the head image -> homography between image pixels and the table plane (metres). numpy + cv2.
Table frame: origin = the mat's near-left corner, x = to the robot's right along the near edge, y = away from the robot."""
import cv2
import numpy as np

MAT_HSV = ((72, 60, 60), (98, 255, 255))  # OpenCV HSV (H 0-179) range around the mat, sRGB (46, 178, 150) = HSV (84, 189, 178); navy bin H ~104


def mat_mask(img_rgb, hsv=MAT_HSV):
    """uint8 mask (255) of the largest mat-coloured blob, opened / closed with a kernel that scales with the width."""
    m = cv2.inRange(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV), np.array(hsv[0]), np.array(hsv[1]))
    k = np.ones((max(3, round(img_rgb.shape[1] / 256)) | 1,) * 2, np.uint8)
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    return (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8) * 255 if n > 1 else m


def _cross(l1, l2):
    """Intersection of two lines (point, unit direction), None if (nearly) parallel."""
    A = np.array([l1[1], -l2[1]]).T
    if abs(np.linalg.det(A)) < 1e-6:
        return None
    return l1[0] + np.linalg.solve(A, l2[0] - l1[0])[0] * l1[1]


def find_table_quad(img_rgb, hsv=MAT_HSV, tol=2.0, min_frac=0.05, min_support=30, min_inside=0.97):
    """The mat's 4 corners in image pixels (float (4, 2)) ordered far-left, far-right, near-right, near-left, or None.
    The mat is partly hidden (bins on its far side, grippers on the near corners), so the corners are not found directly; the four EDGE LINES are.
    The convex hull of the largest mat blob keeps every visible stretch of each edge: a hidden middle of an edge is bridged ON the edge line, a
    hidden corner is cut by a chamfer that no contour pixel supports. Hull sides on the image border are dropped (the mat may run off the frame).
    A side's support = blob contour pixels within tol px of its line; the 4 best-supported sides with outward normals >= 30 deg apart are the
    edges, each refit (cv2.fitLine) to its support, and neighbours (in normal-angle order) are intersected, so hidden or off-frame corners come
    out of the lines. None when the blob is small, an edge has < min_support pixels, or < min_inside of the blob lies inside the quad."""
    h, w = img_rgb.shape[:2]
    m = mat_mask(img_rgb, hsv)
    if cv2.countNonZero(m) < min_frac * h * w:
        return None
    cnt = max(cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
    pts = cnt[:, 0].astype(np.float64)
    hull = cv2.approxPolyDP(cv2.convexHull(cnt), tol, True)[:, 0].astype(np.float64)
    c = hull.mean(0)
    border = lambda p: (p[0], p[1], w - 1 - p[0], h - 1 - p[1])  # noqa: E731 (distances to the left, top, right, bottom image border)
    sides = []
    for a, b in zip(hull, np.roll(hull, -1, 0)):
        d = b - a
        L = np.hypot(*d)
        if L < 1e-6 or any(max(s, t) <= tol + 1 for s, t in zip(border(a), border(b))):
            continue
        n = np.array([d[1], -d[0]]) / L
        n = n if n @ ((a + b) / 2 - c) > 0 else -n  # outward
        sup = pts[np.abs((pts - a) @ n) <= tol]
        sides.append((len(sup), np.arctan2(n[1], n[0]), sup))
    edges = []
    for s in sorted(sides, key=lambda s: -s[0]):
        if all(abs((s[1] - e[1] + np.pi) % (2 * np.pi) - np.pi) >= np.radians(30) for e in edges):
            edges.append(s)
        if len(edges) == 4:
            break
    if len(edges) < 4 or edges[-1][0] < min_support:
        return None
    edges.sort(key=lambda e: e[1])  # normal angle increasing = clockwise on screen (y down): left, far, right, near
    lines = []
    for _, _, sup in edges:
        vx, vy, x0, y0 = cv2.fitLine(sup.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0], np.float64), np.array([vx, vy], np.float64)))
    f = int(np.argmin([abs((e[1] + 1.5 * np.pi) % (2 * np.pi) - np.pi) for e in edges]))  # far edge: outward normal nearest straight up (-90 deg)
    q = [_cross(lines[(f + i - 1) % 4], lines[(f + i) % 4]) for i in range(4)]  # far-left, far-right, near-right, near-left
    if any(p is None for p in q):
        return None
    q = np.array(q)
    quad = np.zeros_like(m)
    cv2.fillConvexPoly(quad, np.round(q * 16).astype(np.int32), 255, cv2.LINE_8, 4)
    if cv2.countNonZero(cv2.bitwise_and(m, quad)) < min_inside * cv2.countNonZero(m):
        return None
    return q


class TableGrid:
    """Homography (cv2.findHomography) between head-image pixels and the table plane: table frame origin = the near-left corner, x along the
    near edge to the robot's right, y away from the robot (m). corners_px ordered far-left, far-right, near-right, near-left (find_table_quad)."""

    def __init__(self, corners_px, table_w_m=1.11, table_d_m=0.685):
        self.corners, self.w, self.d = np.asarray(corners_px, np.float64).reshape(4, 2), table_w_m, table_d_m
        self.H = cv2.findHomography(self.corners, np.array([[0, table_d_m], [table_w_m, table_d_m], [table_w_m, 0], [0, 0]], np.float64))[0]
        self.Hinv = np.linalg.inv(self.H)

    @staticmethod
    def _map(H, a, b):
        p = H @ np.stack([np.asarray(a, np.float64), np.asarray(b, np.float64), np.ones(np.shape(a))])
        return p[0] / p[2], p[1] / p[2]

    def to_table(self, u, v):
        """Pixel (u, v) -> table (x, y) m (scalars or same-shape arrays)."""
        return self._map(self.H, u, v)

    def to_pixel(self, x, y):
        """Table (x, y) m -> pixel (u, v)."""
        return self._map(self.Hinv, x, y)

    def draw(self, img, step_m=0.05, color=(255, 255, 255), edge=(255, 210, 0)):
        """Copy of img (RGB, the corners' image) with the table outline and a step_m grid; each line labelled x<cm> / y<cm> where it first enters
        the image (lines along y from the near edge, lines along x from the left edge)."""
        out, (h, w) = img.copy(), img.shape[:2]
        P = lambda x, y: tuple(int(round(float(t))) for t in self.to_pixel(x, y))  # noqa: E731

        def label(xs, ys, text):
            for u, v in zip(*self.to_pixel(xs, ys)):
                if 4 <= u < w - 40 and 14 <= v < h - 4:
                    for col, th in (((0, 0, 0), 3), (color, 1)):
                        cv2.putText(out, text, (int(u) + 3, int(v) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, th, cv2.LINE_AA)
                    return

        t = np.linspace(0, 1, 60)
        for x in np.arange(0, self.w + 1e-9, step_m):
            cv2.line(out, P(x, 0), P(x, self.d), color, 1, cv2.LINE_AA)
            label(np.full_like(t, x), t * self.d, f"x{x * 100:.0f}")
        for y in np.arange(0, self.d + 1e-9, step_m):
            cv2.line(out, P(0, y), P(self.w, y), color, 1, cv2.LINE_AA)
            label(t * self.w, np.full_like(t, y), f"y{y * 100:.0f}")
        cv2.polylines(out, [np.round(self.corners).astype(np.int32)], True, edge, 2, cv2.LINE_AA)
        return out
