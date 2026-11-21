# Shipping labels

Fictional "LabelLoop Express" labels. The QR payload `420/<ZIP5>/<seq3>` is all the robot reads (routes on ZIP3 via `SORT_PLAN`);
the rest (FROM / SHIP TO / tracking) is realism. The SAME artwork (`sim/make_qr_boxes.label_image`) is the sim parcel texture and the print.

- `labels.csv`: every label: id, payload, ZIP, lane (A blue bin / B red bin / reject black bin), target bin, recipient, address, tracking, size, sim id, real item.
- `labels_letter.pdf`: US Letter, 300 dpi, true size, grey cut lines, 50 mm ruler per page.
- `preview_large.png` (100 x 70 mm), `preview_small.png` (62 x 62 mm).

16 labels: 10 for items (package 1 -> 95112, package 2 -> 90210, package 3 mailer -> 10001, the four sim-only boxes 94110 / 90015 / 60601 / 95014
"not yet bought", the flat retail boxes Bhindi LL-08 -> 95126, LOBAR LL-11 -> 90024, Miro LL-14 -> 02139) + 6 spares (2 lane A, 2 lane B, 2 reject).

Registry (single source): `sim/qr_sort_task/props.py` `LABELS`; lanes from `SORT_PLAN` (same copy `build_task.py` uses). Regenerate:

    .venv/bin/python sim/make_labels.py        # writes docs/labels/*, asserts every payload decodes from the rendered pages

Print: PDF viewer, scale **100% / Actual size** (not "fit to page"), check the 50 mm ruler with a ruler, cut on the grey lines,
stick label up, flat, on any large face (the robot turns the parcel; keep the QR clear of tape and creases). Small labels are for
the small boxes (76 mm cube, white retail box). Keep the QR white quiet zone (label margin) intact.
