"""Find the green work mat in a head image, draw the table grid (labelled 5 cm lines) and print the corners.
usage: python scripts/table_grid_preview.py IMG [--w 1.11 --d 0.685] OUT.png"""
import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from labelloop.perception.table_grid import TableGrid, find_table_quad  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("img")
    ap.add_argument("out")
    ap.add_argument("--w", type=float, default=1.11, help="table width (m), along the near edge")
    ap.add_argument("--d", type=float, default=0.685, help="table depth (m), away from the robot")
    a = ap.parse_args()
    bgr = cv2.imread(a.img)
    if bgr is None:
        sys.exit(f"cannot read {a.img}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    q = find_table_quad(rgb)
    if q is None:
        sys.exit("no table quad found (mat colour range, too little mat visible, or an edge off the image)")
    for name, (u, v) in zip(("far-left", "far-right", "near-right", "near-left"), q):
        print(f"{name:>10} ({u:7.1f}, {v:7.1f})")
    cv2.imwrite(a.out, cv2.cvtColor(TableGrid(q, a.w, a.d).draw(rgb), cv2.COLOR_RGB2BGR))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
