# Tasks for Pratham's coding agent (strict scope)

You are a coding agent helping the LabelLoop team at the hackathon. Work ONLY on the tasks below.

## Rules (must follow)
- Work on a new branch `pratham/<task>` from the latest `main`; push the branch; do NOT push to `main`, do NOT merge, do NOT force-push. Ayush's agent reviews and merges.
- Code + tests only. Do NOT touch the robot (no SSH to 192.168.1.21, no WebUI runs), do NOT call any WALLX endpoint, do NOT read or print `.env` or any key.
- Only create/edit the files listed per task. Do not edit `labelloop/harness.py`, `labelloop/brain/state_machine.py`, `labelloop/robots/event_server.py`, `configs/*.yaml` (another agent is editing them live).
- Terse Python, numpy/cv2 only (already in the venv). Run `.venv/bin/python -m pytest -q` (Windows: `.venv\Scripts\python -m pytest -q`); the full suite must stay green.
- When done: one commit per task with a clear message, push the branch, and write a 5-line summary at the end of this file under "Status".

## Task 1: table grid (pixel <-> table centimetres)
New file `labelloop/perception/table_grid.py` + `tests/test_table_grid.py`.
- `find_table_quad(img_rgb)`: find the green work mat (HSV range around the mat colour, sRGB ~ (46, 178, 150); make the range a parameter) in a 1280x720 head image; the mat is partly occluded by bins (top) and grippers (bottom corners), so fit the four table EDGE LINES (cv2.HoughLinesP or line fits on the mask contour) and intersect them to get the 4 corners even when corners are hidden. Return 4 corners in image pixels ordered far-left, far-right, near-right, near-left.
- `TableGrid(corners_px, table_w_m=1.11, table_d_m=0.685)`: homography (cv2.findHomography) between the image and the table plane in a table frame (origin = near-left corner, x = to the robot's right along the near edge, y = away from the robot, metres). Methods `to_table(u, v) -> (x, y)` and `to_pixel(x, y) -> (u, v)`, `draw(img, step_m=0.05)` returning a copy with a labelled 5 cm grid.
- Script `scripts/table_grid_preview.py IMG [--w 1.11 --d 0.685] OUT.png` that finds the quad, draws the grid and prints the corners.
- Tests: render a synthetic image (green quad warped by a known homography, with rectangles occluding two corners), check corners within 4 px and round-trip error < 5 mm.

## Task 2: replay checker uses the real profile
Edit only `scripts/replay_event_images.py` and `tests/test_replay_event_images.py`.
- Add `--profile real|sim` (default real): load `configs/task_qr_sort.yaml`, apply the profile's `bin_rois` / `induction_roi` overrides (see `profiles.real` in the yaml) before checking, so bin blanking uses the real bin regions.
- Add `--grid` to overlay the Task 1 grid on the annotated images (skip if Task 1 isn't merged yet).

## Status
- Task 1 `pratham/table-grid` (3 new files): `labelloop/perception/table_grid.py` `find_table_quad` (largest blob in `MAT_HSV` -> convex hull -> the 4 best-supported edge lines, refit with cv2.fitLine and intersected, so corners hidden by bins / grippers or off the frame come out of the lines; None unless >= 97% of the mat lies inside the quad) + `TableGrid` (findHomography, `to_table` / `to_pixel`, `draw` = labelled x/y 5 cm grid); `scripts/table_grid_preview.py IMG [--w --d] OUT.png`.
- Task 1 tests (8): mat warped by a known homography, bins on the far edge, a parcel, two corners occluded (near pair / far-left + near-right; also near corners off-frame): corners <= 0.9 px (limit 4), round-trip <= 1.1 mm (limit 5); 1.8 px with noise + JPEG 80. Real frame `tests/data/real_head_qr.jpg`: the mat fits `MAT_HSV` (H 86-88, S 165-178) and the far / left / right edges are found, but the NEAR edge is below the image, so it returns None (safe: the only 4th candidate, the black-bin chord, has 20 px support); a grid on that view needs a 4th constraint (head intrinsics + table depth, or a measured near-edge line), not in this spec. docs/AGENT_CONTEXT.md gives the real table as 1.17 x 0.94 m: pass `--w 1.17 --d 0.94` (defaults kept at this spec's 1.11 x 0.685).
- Task 2 `pratham/replay-profile`: `--profile real|sim` (default real) applies the profile's `bin_rois` / `induction_roi` and also `whole_table` (as `harness.apply_profile`; without it real frames get the sim camera's projected-tabletop mask); `--grid` draws the Task 1 grid (TableGrid defaults) under the annotations, prints the corners, and skips with a note while `table_grid` is not merged.
- Task 2 tests (+4): overrides equal the yaml; a code in the real red bin / the sim tray flips accepted vs ignored between the profiles; `--grid` without `table_grid`; grid overlay (skipped on its own branch, passes with Task 1 merged).
- Suite (Windows, both branches on ed9e3a0): table-grid 183 passed / 2 skipped, replay-profile 178 / 3, both merged 187 / 2, each + 1 failure that also fails on unmodified main: `test_real_profile::test_empty_ref_is_the_initial_reference` (Windows only: a '/' suffix asserted on a '\' path). Also pre-existing on Windows: `test_wallx_local::test_local_translates_both_ways_and_logs_latency` asserts latency > 0 but the monotonic clock ticks every 15.6 ms (6/20 fail in isolation on main). Both untouched (out of scope). Both branches carry this same Status, so they merge cleanly.
