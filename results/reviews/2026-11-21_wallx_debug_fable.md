**Ranked likely causes — what *we* add or drop in the relay**

1. **No `session_id`.** Best evidence in the packet: identical requests → different outputs (max 0.17 rad) despite seed replay. That only happens if hidden server-side state exists and we don't key it. `server_auto_history` declares `session_id_key`, and 4 robots are active on one box. Fix: send `session_id: "wallx-<robot_uuid>-<episode_uuid>"`, stable across every call within an episode, with `use_history: true`.

2. **No `control_context`, while running sync.** 32 × 0.0337 s = **1.08 s** of action per call but **~1.4 s** round trip — the call outlasts its own chunk. The model is trained with RTC (`future_delay_max 4`) and with `eff_delay_ms`/`grip_delay_ms`/`z_offset_mm` as trained inputs; sending none makes it plan as if delay ≈ 0. Symptom fits perfectly: overshoot into bins, wrong arm, "went past the parcel". Fix: populate `control_context` (values are −1..1 — get the normalization from the server owners) with the measured 1.4 s, `z_offset_mm: 0`; and/or switch to async chunk streaming so chunks overlap instead of sync.

3. **`reset_history: true` on every sub-goal and retry.** `L_max 4` at 32 Hz is only ~125 ms, and we wipe it at exactly the moments it matters. That is the "let go of the held box, put it in the nearest bin" and "keep grabbing a bin with the parcel already placed" behaviour — the model re-grounds from scratch each time. Fix: reset only at episode start; a sub-goal change should be a new `instruction` on the same session, nothing else.

4. **Instruction leaves the referent and arm open.** "pick up the box" against three box-shaped props is genuinely ambiguous, and nothing binds an arm. Fix: always name referent + arm — "with the left arm, pick up the IQBAR box" — and have the relay reject bare "the box".

5. **Relay rewriting the chunk (`--lock-arm`, `--open-wide`).** RTC assumes the previous chunk was executed; substituting 32 rows breaks that assumption and moves the gripper out of its trained distribution. Fix: stop splicing rows — hold the idle arm at the client or suppress it before the chunk, and pre-open the gripper as its own step instead of clamping to 0.85.

6. **Nothing terminates the episode.** After a successful placement the instruction keeps running, so the policy re-grounds and picks a bin. Fix: relay-side completion check (gripper open + EEF inside the target bin volume for N frames) → stop sending, don't let the instruction "keep going".

7. **Sub-goals outside the action space.** "turn the box so its other side faces the camera" isn't pick-and-place; task B training is pick/place/move. Fix: decompose only into pick / place / move-to; do reorientation with a scripted in-hand rotate, not WALLX.

8. **Grasp geometry, not policy.** IQBAR is **102 mm** wide; the gripper opens **97.5 mm**. A lateral grasp on the flat face cannot close — that's why flat/left-to-right failed and narrow-side front-to-back gave 5/6. Fix: stage IQBAR narrow-side-up, grasped across the 64 mm dimension; never plan a 102 mm span.

**Run first:** the `session_id` determinism test — 10 minutes, and it decides whether 2–6 are even measurable. Send one identical request twice (a) exactly as today, (b) with a fresh fixed `session_id`, against a warm server. If (b) collapses the 0.17 rad spread to ~0 and (a) doesn't, shared hidden history is confirmed; re-baseline every other number only after that fix is in.