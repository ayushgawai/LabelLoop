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
(write here when done)
