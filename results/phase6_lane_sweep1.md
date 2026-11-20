# Phase 6 sweep 1: lane primitives, WALLX passthrough (2026-11-19)

Scene: lane cases 7-11 (blue mat front-left, orange mat front-right, floor reject basket off the right edge).
WALLX gets one fixed instruction for the whole 90 s episode (no harness). 2 episodes per line. Score 8 = parcel on target.

| case | instruction | ep0 | ep1 |
|---|---|---|---|
| 7 (ZIP A) | push the block onto the blue mat | 0 | 0 |
| 7 | put the block on the blue mat | 0 | **8** |
| 8 (ZIP B) | push the block onto the orange mat | 0 | 0 |
| 8 | push the block onto the yellow mat | 0 | 0 |
| 9 (unknown ZIP) | put the block into the basket on the right | 0 | 0 |
| 9 | drop the block off the right side of the table | 0 | 0 |
| 11 (garbage) | put the red block into the basket on the right | 0 | 0 |

What the videos show:
- **Invalid placement.** Every parcel teleported to the same spot at the far side of the table, between the two mats, instead of into the induction zone in front of the arms. The arms mostly hovered near home. This sweep does not fairly test pushing.
- **Pick-and-place still works.** "put the block on the blue mat" picked up the parcel and placed it on the mat once (1/2), even from that far spot.
- **"push" did not produce a push** in 4/4 episodes.
- **The floor basket is outside every camera view.** WALLX never acted toward it (0/6).

Next steps:
- Fix the arrival position and randomize it per episode.
- Re-test push and move wordings.
- Compare the floor basket with an on-table reject bin (visible, and inside the rules' "standard workbench" envelope).
