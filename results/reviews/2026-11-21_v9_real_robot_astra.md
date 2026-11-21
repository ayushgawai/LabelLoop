**NO-GO for autonomous sorting with v9 as configured.** First target: one known top-label parcel, head-only decoding, one supervised WALLX place. Unread, oversize, or ambiguous parcels should stop the run, not trigger a fallback motion. Line references below are approximate.

### Blockers

1. **`harness.py:139–165, 176–181`; `task_qr_sort.yaml:75–86, 149–158` — wrong geometry for both motion and perception.** The fit, held-QR projection, table mask and push limits use sim geometry; the configured far-row bins are physically in front. A plausible fit can put fingertips into a bin, accept a binned QR, or verify the wrong destination. Load real intrinsics at the decoded image resolution and compose the left-base↔head transform with the measured robot frames; verify transform direction against physical landmarks. Measure table height, edge and bin footprints. For the first scan, use measured image masks without metric projection.

2. **`state_machine.py:112–122, 174–182` — disabling retries does not disable first motion.** `_route` selects SWIPE for floor rejects and LIFT2 for `big` routed parcels without checking their try limits. `tries.SWIPE: 0` or `tries.LIFT2: 0` is **not** an interlock. Add a latched operator-hold state before dispatch for every disallowed action; test unknown ZIP, missing label and oversized parcel. Don’t switch to `tray` unless a real tray exists.

3. **`harness.py:50–62, 168–193` — wrist QR bypasses bin exclusion.** `decode()` contributes wrist text directly to `seen`; two views of a bin label can route the active parcel. Disable wrist decoding entirely. For a strict “never read” rule, blank measured bin regions **before** full-frame and crop decoding, not just after `locate()` via a centre test. Hold on conflicting parcel codes.

4. **`harness.py:41–43, 323–365`; `robots/event_server.py:64–70` — arbitrary start pose becomes home.** A goal-text change resets the harness; the next observation can capture mid-reach as “home.” Scripts then command the idle arm there, and waypoint/home completion is counted in calls, not measured arrival. Require explicit episode arming at verified park poses; reject mid-motion resets. Advance on fresh pose/gripper feedback within tolerance, with timeout to stop.

### Should-fix before unattended runs

5. **`harness.py:63–67, 153–165`; `state_machine.py:250–260` — background can absorb the parcel.** Startup captures whatever is present; failed verification explicitly rebases over a stuck object. Also, a parcel above the 1% presence threshold but below the 2% whole-table fit gate can trigger SCAN yet never get a fit. Require operator-confirmed empty baseline after failures; tune both gates using the smallest real parcel at table corners.

6. **`task_qr_sort.yaml:29–30`; `state_machine.py:72–82, 166–172` — budget is simulated, not elapsed.** The 1,500-s case and 11-s WALLX estimate don’t represent the 480-s event or LAN backend; 0.07 s inference isn’t motion completion either. Use a monotonic case deadline and measured end-to-end chunk times, including execution and settling.

7. **`state_machine.py:84–97`; `task_qr_sort.yaml:100–113, 140–142` — bottom-label search has no reliable success path.** Config notes bottom SHOW is unreachable; in-plane SPIN cannot reveal the underside and INSPECT has zero tries. Report *unread* and request a manual flip for initial runs.

### Five-line go/no-go checklist

1. Test physical stop and link-loss stop; clear both swept volumes, park both arms, and keep hands out after arming.
2. Disable SHOW/SPIN/SLIDE/LOOK, calibration motion, wrist reads and retries; dry-test hard holds for SWIPE/LIFT2 and failed placement.
3. On stationary images verify empty reference, smallest parcel across the table, correct ZIP3 lanes, and zero bin-code reads with both bins populated.
4. Verify G embodiment, opening/feedback units, both arm frames and safe home path at reduced speed; check landmark reprojection before any metric script.
5. Set `max_parcels: 1`; supervise one narrow top-label place and verify the actual bin and empty table before rearming.