# Why does WALLX get confused for us? (review packet, 2026-11-21 18:05)

Other teams say WALLX follows their instructions well. For us it often grabs the BINS instead of the parcel, ignores which arm we
name, and cannot do non-pick-and-place sub-goals. Find what WE do wrong. Answer in <= 30 lines: ranked likely causes with a concrete
fix each (exact request field / relay change / setup change), then the single experiment to run first.

## What reaches WALLX (our relay sits between the organisers' robot client and WALLX; code: labelloop/robots/event_server.py, labelloop/wallx_client.py)
- Organisers' client: start_desktop.py --robot-mode end_pose --infer-type sync --embodiment teletwin_desktop/gripper-g
  --image-wire-format base64 --attach (default infer params preset wallx_2606_ex001_g_rtc; we run sync). ~1.4 s per call end to end.
- Request (public_v2) forwarded as the client built it, except: `instruction` replaced by ours; `reset_history: true` added on the
  first call and on every sub-goal change and retry. We do NOT send `session_id`, `use_history`, `control_context`, `normalization`.
- Images: front 1280x720, wrists 640x480, base64 JPEG as sent by the client (re-encoded only when an overlay is on; recent runs: off).
- State: left_ee / right_ee position (m), rotation_rpy, gripper; home-relative (~0 at rest), as the real client sends it.
- Reply: 32 rows x 14 (Rel EEF, 0.0337 s/row). Relay may modify it: `--lock-arm` replaces the idle arm's 32 rows with a hold of its
  current state; `--open-wide 0.85` sets gripper rows to >= 0.85 when the chunk opens the gripper.

## What the server says (metadata)
model wall-x 2610-dualhead-v2, checkpoint 0_130000, task "B", env x2robot, robot_type desktop, dataset_family ex_normal,
system prologue "Embodiment: X2 EX001 Desktop; Camera Setup: front view, left wrist view, right wrist view; Frequency: 32HZ;
Action Space: Rel EEF"; template_revision r11-training-c0ac722-rd-umi32-v2. cb_history: L_max 4, hist_lens [0,2,4],
server_auto_history {enabled, default_use_history: true, request_key use_history, reset_key reset_history, session_id_key session_id}.
cb_rtc: training_time, future_delay_max 4. control_context: only eff_delay_ms, grip_delay_ms, z_offset_mm are trained (values -1..1).
sampling: request_sha256 seed replay (but identical requests gave different outputs: max 0.17 rad). serving: 8 robots max, 4 active.

## What we observed (real robot)
- Bins are open shoe boxes (navy = blue bin back-left, red Nike = red bin back-right, black/white box = black bin at the right edge).
  Parcels: IQBAR 267x102x64 mm, Miro 216x165x38, Bhindi 165x102x38; gripper opens 97.5 mm.
- "pick up the box": the right arm went to the black bin. "pick up the IQBAR box": left arm to the IQBAR, right arm dragged the red bin.
- Idle arm locked (relay): IQBAR standing on its narrow side, front-to-back: 5/6 delivered ("put the IQBAR box into the blue/red bin").
  Lying flat (102 mm wide) or left-to-right: not grasped / knocked over. Two-arm "squeeze the short ends ... lift" lifted it twice.
- "turn the box you are holding so its other side faces the camera": WALLX dropped it into the nearest bin instead.
- With the parcel already in a bin and the instruction still running, WALLX goes on to grab a bin.
- Organiser sample task wording: "put the box into the blue bin".

## Our hypotheses (check / rank / add)
1. No `session_id`: server auto-history may not work per robot, or mixes with other robots (4 active robots on the server).
2. `reset_history: true` too often (every sub-goal / retry) -> no temporal context, jittery decisions.
3. Our shoe-box bins look like graspable "boxes"; training props (task B) were probably different bins / trays.
4. Arm lock / open-wide break the model's expectation that its last chunk was executed (history mismatch).
5. Instruction wording far from training phrasing.
