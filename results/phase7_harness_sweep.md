# Phase 7: full harness on the row layout, cases 18-23 (2026-11-20)

Scene: three bins in a row on the far side (orange left, blue centre, black reject right) and one parcel dropped in the
induction zone at t = 6 s. The full harness runs: SCAN, then PLACE or REJECT, then HOME, then VERIFY, retrying up to 3 tries.
Each episode is 150 s. The score is sim-scored: sort 8 (object in the correct bin) plus return 2 (joints near default).

| case | parcel | correct target |
|---|---|---|
| 18 | ZIP 95112 (lane A) | blue bin |
| 19 | ZIP 90210 (lane B) | orange bin |
| 20 | ZIP 10001 (unknown ZIP) | black bin |
| 21 | kraft box, no label | black bin |
| 22 | red cube (garbage) | black bin |
| 23 | the five above, 150 s apart | each to its bin |

## Results

| batch | change since previous | episodes | sorted | returned home | routing correct |
|---|---|---|---|---|---|
| 1 | first run | 10 | 2 | 0/2 | 10/10 |
| 2 | scripted HOME; drop spot narrowed to just in front of the right gripper | 15 | 5 | 4/5 | 15/15 |
| smoke | Policy Space `exclusive` session mode (harness reset each episode) | 2 (case 22) | 2 | 2/2 | 2/2 |
| 3 | none (first fully clean batch) | 30 | 8 | 8/8 | 29/30 |
| 4 (stopped after 18 of 32: the scene changed to the real props) | reject verified in the black bin; 2x ROI QR re-decode | 18 (cases 18-20) | 7 | n/a | n/a |

Batch 4 by case, sorted out of 6: case 18 blue 4, case 19 orange 1, case 20 unknown ZIP 2.

Batch 3 by case, sorted out of 6:

| case | 18 blue | 19 orange | 20 unknown | 21 no label | 22 garbage |
|---|---|---|---|---|---|
| sorted | 1 | 2 | 1 | 1 | 3 |

## Findings

- **Perception and routing work.** 56 of 57 parcels went to the right target. The one miss was a 40 mm QR that failed to decode
  on six identical static frames, so it was routed as "no label". Batch 4 adds a 2x re-decode of the ROI crop when the read misses.
- **WALLX's grasp is the bottleneck.** Batch 1 grasped only the parcel that landed nearest the right gripper. Centre drops got
  no real reach. Narrowing the drop spot to lateral +15 to +23 cm raised the sort rate from 20% to about 30%.
- **WALLX does not return home on request.** The "return" term was 0/2 until HOME was scripted: the robot now eases back
  to the episode-start pose without a WALLX call. After that it was 14/15. It can be switched off with `scripted_home: false`
  if the event rules require every motion to come from WALLX.
- **Batches 1-2 were partly invalid.** With `session_mode: multiplexed_serial`, Policy Space never calls
  `model.initialize_episode`. The harness and WALLX history carried over between episodes, so only episode 0 of each run
  was clean. This is fixed (`exclusive`).
- **Reject false positives.** In 6/30 batch-3 episodes the harness counted a reject as done because the object left the
  zone, but it landed outside the black bin. The harness then stopped retrying. The reject check now needs a change inside
  the black-bin box against the snapshot taken when the object arrived.
- **Tries land late.** Successes came at 22-119 s into a 150 s episode. The 3-try budget is what makes a ~25%-per-try
  policy reach about 30% per parcel.

## Next

This layout is retired. The scene now uses Pratham's measured props (shoe-box bins, real packages) with blue + red bins in front, the black bin beside the right arm, parcels swiped in from the left, and one label on a random face (scene v2, cases 18-30).
