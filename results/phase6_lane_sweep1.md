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

## Sweep 2: parcels arriving in front of the arms, random per episode (fixed scene, 2 episodes each)

| case | instruction | score |
|---|---|---|
| 7 (ZIP A) | push the block onto the blue mat | 0, 0 |
| 7 | move the block onto the blue mat | 0, 0 |
| 7 | put the block on the blue mat | 0, 0 |
| 8 (ZIP B) | push the block onto the orange mat | 0, 0 |
| 8 | put the block on the orange mat | 0, 0 |
| 13 (no label) | put the block into the black bin (on-table reject bin) | 0, 0 |
| 14 (garbage) | put the red block into the black bin | 0, 0 |
| 9 (unknown ZIP) | drop the block off the right side of the table (floor basket) | 0, 0 |

0/16. In the videos the arms hover near the parcel and do not grasp or push it.
Conclusions:
- WALLX, zero-shot, has no push/swipe skill in this scene.
- Flat mats are not good targets for it.
- Next: a control run that repeats Phase 4 (case 1, "put the block into the blue bin", 2/2 back then), and lanes built as bins (cases 15-17).

## Sweep 3: lanes as open bins (cases 15/16), plus controls

| case | instruction | score |
|---|---|---|
| 15 (A) | put the block into the blue bin | 0/3 |
| 16 (B) | put the block into the orange bin | 0/3 |
| 13/14 (reject) | put the block into the black bin / gray bin | 0/4 |
| 16 (diagnostic, wallx state logged) | put the block into the blue bin | 0/1 |
| 1 (Phase 4 control) | put the block into the blue bin | 1/2, then **2/10** |

Control map (case 1, 10 episodes, the cube spawns anywhere on the table):

| cube position (head image) | arm that went for it (wire log) | result |
|---|---|---|
| left (ep2, 4, 5, 6) | left arm | 0/4 (reaches the cube, never grasps) |
| centre / right (ep0, 1, 3, 7, 8, 9) | right arm | 2/6 |

Findings:
- WALLX zero-shot in X2Real is about 20% on the simplest pick-into-bin task. Phase 4's 2/2 was luck.
- Every success so far is the right arm. The left arm reaches but approaches with a twisted gripper.
- Pose conversion checked against X2Real source:
  - The wire is EE minus start EE, in the robot root frame, with R_rel = R R_ref^T as Euler XYZ (`wallx_follow_from_init_ee_root_local_mimic_gripper`). pose.py inverts this exactly.
  - Both arm bases are mounted rpy 0 at y +-0.266 (URDF), so conversion is symmetric for the two arms. The left-arm failure is not a frame bug in our code.
  - Head camera: robot-left arm = image-left. WALLX also acts that way: it uses left_ee for image-left cubes.
- Open question for event day: the cloud API's absolute base frame. Its example state is around (0.29, +-0.03, 0.38) vs our home (0.1185, 0, 0.181). The API computes deltas internally, so an offset mostly shifts proprioception. Read one real EndPose at home on the day and update configs/ex001_6r.yaml.
- Image resize matches stock X2Real (1280x720 -> 640x480, INTER_AREA).

Decision: build the demo around what scores.
- Pick-and-place, not swipe.
- Three bins in a row on the far side, at the case-1 positions where WALLX succeeds.
- Parcels arrive centre-right, in front of the right arm.
