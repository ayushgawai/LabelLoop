# WALLX prompt sweep · scenario 5

10 scored Windows X2Real episodes. All used LAN **wall-x 2610-dualhead-v2** with head and both wrist cameras. **0/10** lifted the IQBAR box and settled it fully inside zone 2.

| Rank | Strategy and trial evidence | Open | Grasp | Lift | In zone 2 | Literal | Calls to success |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | Object description [02](trial_02.json) | 1/1 | 0/1 | 0/1 | 0/1 | 0/1 | — |
| 2 | Three fresh-history rounds [09](trial_09.json) | 1/1 | 0/1 | 0/1 | 0/1 | 0/1 | — |
| 3 | Atomic sequence [04](trial_04.json) | 1/1 | 0/1 | 0/1 | 0/1 | 0/1 | — |
| 4 | Named arm and grip [06](trial_06.json) | 1/1 | 0/1 | 0/1 | 0/1 | 0/1 | — |
| 5 | Detailed one-shot [03](trial_03.json) | 1/1 | 0/1 | 0/1 | 0/1 | 0/1 | — |
| 6 | State and negation [07](trial_07.json) | 1/1 | 0/1 | 0/1 | 0/1 | 0/1 | — |
| 7 | Short one-shot [01](trial_01.json) | 1/1 | 0/1 | 0/1 | 0/1 | 0/1 | — |
| 8 | Pick then place [05](trial_05.json) | 1/1 | 0/1 | 0/1 | 0/1 | 0/1 | — |
| 9 | Container grammar [08](trial_08.json) | 1/1 | 0/1 | 0/1 | 0/1 | 0/1 | — |
| Probe | Displacement with history reset [10](trial_10.json) | 1/1 | 0/1 | 0/1 | 0/1 | 0/1 | — |

Ranked by pickup and placement, then lift, grasp, placement, literal completion, and median closest estimated jaw distance.

## Scenario 5 wording

**Object description** had the strongest observed approach. The sweep has yet to demonstrate a full pickup and placement with this wording.

> Put the dark-brown IQBAR carton, about 20 by 10 by 5 centimeters, standing on its long edge near the center of the table, into zone 2.

## Displacement and transfer

- Trial 10 never met the approach trigger, so no teleport occurred. [Trial evidence](trial_10.json)
- The real 90-call scenario 5 recipe used all three cameras, arm locking, slower actions, wide open and deeper closing. It logged one pickup and placement across four recipe runs, and no recovery in two human-move runs. [Real scenario logs](../../scenarios/RESULTS.md)
- The simulated carton was 20 × 10 × 5 cm on a fixed layout. The prop notes list the physical IQBAR at 26.7 × 10.2 × 6.4 cm. The simulator used the EX001-6R G embodiment and different action timing. Cloud was unused. [Prop measurements](../../docs/props/README.md)
- Grasp and lift were confirmed by the nearby closing jaw and at least 1.5 cm of box clearance across two samples. A stationary grip without visible confirmation can be missed. The entire box footprint had to settle inside the tape. Each linked JSON has calls, measurements, and head-frame paths.
