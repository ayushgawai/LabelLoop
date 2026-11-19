# Phase 1 gate: LabelLoop passthrough in X2Real (2026-11-19, Windows RTX 5080 box)

LabelLoop Policy Space server, `LABELLOOP_PASSTHROUGH="put the fruits into the basket"`, cloud WALLX,
`artixon_arm6a_G_pick_fruits_into_basket_id_env`, 5 episodes, 50 s sim each (~31 WALLX calls/episode).

| Episode | Score /10 | Failed stage | Wall s |
|---|---|---|---|
| 0 | 0 | place_fruit_1 | 387 |
| 1 | 0 | place_fruit_1 | 420 |
| 2 | 0 | place_fruit_1 | 421 |
| 3 | 6 | place_fruit_1 | 451 |
| 4 | 5 | place_fruit_2 | 445 |

160 WALLX calls, 0 errors, 6–10 s per call. Video frames: arm reaches a fruit in 4/5 episodes (one hovered at the basket).
Gate (correct arm to a fruit in 3/5): **passed**. Supports the per-arm home pose in configs/ex001_6r.yaml.
