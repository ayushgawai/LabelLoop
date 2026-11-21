# Task: visual prompt overlay on the image WALLX sees

WALLX (a vision-language-action model) gets the head image + one short instruction per call. It keeps grabbing the shoe-box
BINS instead of the small parcel. We want to mark the target ON THE IMAGE we forward to WALLX (only the forwarded copy; the
harness and the recorder keep the raw frames).

## 1. `labelloop/perception/overlay.py` (new)
`draw(img_rgb, mode, parcel=None, target=None, bins=None, roi=None, cells=(8, 5)) -> new RGB image` (never modifies the input).
- `parcel`, `target`: pixel boxes (x0, y0, x1, y1) or None; `bins`: {name: pixel box} of all bins; `roi`: pixel box of the free table.
- mode "none": return the input unchanged (same object is fine).
- mode "parcel": thick (6 px) magenta (255, 0, 255) rectangle around `parcel` padded by 12 px, thick green (0, 255, 0) rectangle
  around `target`, and a yellow arrow (cv2.arrowedLine, 5 px) from the parcel centre to the target centre. No text.
- mode "mask": like "parcel", plus every bin in `bins` other than `target` filled with the table green (46, 120, 100) so it
  disappears.
- mode "grid": like "parcel", plus a numbered grid over `roi` (cells cols x rows, thin white 1 px lines, cell number 1..N
  row-major from the top-left in small white text).
- Missing parcel/target: draw what is available.

## 2. Harness (`labelloop/harness.py`): ONLY add, do not change existing logic
- In `step()`, remember the front-camera position of the first accepted head code this call: `self.code_px = codes[0][1]` if
  codes else keep the previous value; reset it to None in `reset()`.
- `prompt_boxes(shape)` -> `(parcel_box, target_box, bins, roi)` in pixels for an image of `shape` (h, w, ...):
  parcel = `self.obj_box` if set, else a 140 x 100 px box centred on `self.code_px`, else None;
  target = the bin box of `self.cfg["lanes"].get(self.sm.lane)` (black for a reject / unknown lane) from
  `self.cfg["scene"]["bin_rois"]` converted with `scene._frac_box(img_like, box)` (or the same maths), None if no lane yet;
  bins = all bin_rois in pixels; roi = `scene.roi_box` equivalent of `induction_roi` in pixels.

## 3. Relay (`labelloop/robots/event_server.py`)
- Flag `--overlay {none,parcel,mask,grid}` (default none), passed to `Relay(..., overlay=...)`.
- In `handle_wallx`, right before `wallx.infer(req)` and only when a harness exists and overlay != "none": decode the front
  image, `overlay.draw(...)` with `harness.prompt_boxes(img.shape)`, re-encode JPEG quality 85, base64 into
  `req["observation"]["images"]["front"]` (keep the same image wire format as the request: base64 string). If the recorder is
  on, also write the overlaid frame as `frames/<index>_front_overlay.jpg` (add a small `TrialRecorder.extra_frame(index, name, jpeg_bytes)`).
- Never fail the relay because of the overlay: on any exception forward the raw image.

## 4. Tests
`tests/test_overlay.py`: draw on a synthetic 720x1280 image for each mode (pixels inside the target box edge are green, parcel
edge magenta, masked bin filled, input unchanged). Relay test: with a fake harness exposing `prompt_boxes`, `--overlay parcel`
changes the front image forwarded to a fake WALLX client and `none` forwards it byte-identical (see tests/test_event_server.py
for the fakes). Full suite must stay green: `.venv/bin/python -m pytest -q`.
