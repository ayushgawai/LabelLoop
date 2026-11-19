# Phase 4 case 1 + Level-3 wording sweep (2026-11-19, 100x100x50 mm boxes)

Full harness, case 1, 1.9 episodes (run cut short): 105 WALLX calls, 0 errors, 0 sorted.
- Bug: 45 s *wall* state timeouts expired after ~5 calls (8-10 s each) -> budgets now in model calls.
- Bug: idle gripper plateau 0.12-0.25 read as a grasp -> grasp now requires open (>=0.45) then steady close.
- LOOK worked twice: right wrist confirmed `LL|ZONE-A|001` (calls 63, 79).

Passthrough sweep, case 1, 2 episodes each, score /8 (sort) + /2 (return):

| Instruction | Score | Behaviour (video) |
|---|---|---|
| put the box into the blue bin | 0, 0 | arms hover near bins, never approach the box |
| pick up the brown box and put it in the blue bin | 0, 0 | same |
| pick up the box | 0, 0 | same |
| pick up the cardboard box with the QR code and put it into the blue bin | 0, 0 | same |

Root cause: the G gripper opens ~97 mm (finger travel 2 x 48.75 mm, URDF), so a 100x100x50 mm box lying flat
cannot be grasped from above, and WALLX ignores the large slab. Same props were planned for event day.
Decision: parcels become 60 mm kraft cubes with a 40 mm QR (50 mm label); re-run the sweep.
