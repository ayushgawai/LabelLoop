# Media for the demo

## real_robot/  (X Square robot at the event, WALLX = wall-x 2610-dualhead-v2)
Each run comes as `<name>.mp4` (head camera), `<name>_tracked.mp4` (parcel tracking box; scenario 5 also shows the harness 3x3 table grid with the located cell), `<name>_montage.jpg` and `<name>_start.jpg` (start position).

| File prefix | What happens |
|---|---|
| sc1_cap_90291_zone2 | Pincode OCR reads 90291 -> zone 2 -> right arm; WALLX picks the cap and puts it in zone 2 |
| sc1_bag_94043_zone1 | Label -> zone 1 -> left arm; WALLX picks the mailer bag and puts it in zone 1 |
| sc2_miro_zone2 | Harness picks the Miro box; WALLX carries it to zone 2 |
| sc3_bag_flip_zone1 | Harness flips the label-down bag label-up; WALLX picks it and puts it in zone 1 |
| sc4_bhindi_handover_zone1_run1 / run2 | Harness picks + hands over in the air; WALLX (head camera only) puts the box in zone 1 (2/2) |
| sc5_iqbar_pick_place_zone2 | WALLX picks the IQBAR box (deep grasp, claw wide open) and puts it in zone 2 |
| sc5_cap_moved_grid_locate | A person moves the cap; the harness re-finds it on its 3x3 table grid and tells WALLX where it is |

`SCENARIO_SETUPS_1-4.pdf`: parcel + start position + exact WALLX prompt per scenario.

## simulation_prompt_sweep/  (X2Real / Isaac on our RTX box, driven by the SAME event model over the LAN)
10+ episodes, one prompt strategy each, IQBAR box standing in the middle, all three cameras. `ALL_TRIALS_overview.jpg` = one frame per trial; `SOL_REPORT.md` = ranked results.

| # | Strategy | Prompt(s) | Claw open | Grasp | Lift | In zone 2 | Video |
|---|---|---|---|---|---|---|---|
| 1 | Short one-shot | Put the IQBAR box into zone 2. | Y | N | N | N | `sim_trial_01_short_one-shot.mp4` |
| 2 | Object description | Put the dark-brown IQBAR carton, about 20 by 10 by 5 centimeters, standing on its long edge near the center of the table, into zone 2. | Y | N | N | N | `sim_trial_02_object_description.mp4` |
| 3 | Detailed one-shot | Approach the upright IQBAR box in the middle of the table from above with the claw open. Place the jaws on either side of its thin width, across the broad faces, instead of gripping the two short ends. Lower the open jaws past the top edge before closing firmly around the box. Lift the box clear of the table, carry it to the taped square at the back right, lower it inside zone 2, and release it. | Y | N | N | N | `sim_trial_03_detailed_one-shot.mp4` |
| 4 | Atomic sequence | Open the claw and move it directly above the IQBAR box. | Lower the open claw around the IQBAR box, with one jaw on each broad face. | Close the claw around the sides of the IQBAR box. | Lift the IQBAR box straight up without opening the claw. | Keep holding the IQBAR box and carry it above the taped square at the back right. | Lower the IQBAR box inside zone 2 and set it down. | Open the claw to release the IQBAR box and lift the empty claw away. | Y | N | N | N | `sim_trial_04_atomic_sequence.mp4` |
| 5 | Pick then place | Pick up the IQBAR box. | Put the IQBAR box into zone 2. | Y | N | N | N | `sim_trial_05_pick_then_place.mp4` |
| 6 | Named arm and grip | Use the right arm to grip the IQBAR box from above, with the jaws across the two broad sides. Lift it and put it into zone 2. | Y | N | N | N | `sim_trial_06_named_arm_and_grip.mp4` |
| 7 | State and negation | The IQBAR box is not in zone 2 yet. Pick up the box from the middle of the table, then put it down fully inside the taped square at the back right. | Y | N | N | N | `sim_trial_07_state_and_negation.mp4` |
| 8 | Container grammar | Put the IQBAR box into the tray. | Y | N | N | N | `sim_trial_08_container_grammar.mp4` |
| 9 | Three fresh-history rounds | Put the IQBAR box into zone 2. | Put the IQBAR box into zone 2. | Put the IQBAR box into zone 2. | Y | N | N | N | `sim_trial_09_three_fresh-history_rounds.mp4` |
| 10 | Displacement with history reset | Put the IQBAR box into zone 2. | Y | N | N | N | `sim_trial_10_displacement_with_history_reset.mp4` |
| 11 | Object description | Put the dark-brown IQBAR carton, about 20 by 10 by 5 centimeters, standing on its long edge near the center of the table, into zone 2. | Y | N | N | N | `sim_trial_11_object_description.mp4` |

## simulation_early_cases/
Earlier simulation cases (bins layout, Nov 19-20) from the sim preview.
