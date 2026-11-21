# Hard rules implementation status

## Part 1 complete

- Added opt-in event-fast PICK → PLACE delivery. PICK uses two consecutive readings of a nonempty close after opening the lane gripper, or reduced presence against the empty-table reference in the original pickup spot. PICK inherits PLACE timeout and retry limits unless overridden.
- The harness selects the front-camera mask. PICK hides every bin and PLACE exposes only the destination, including the black reject bin. The relay holds if a required mask fails to encode. The existing overlay modes still work.
- Nine targeted tests passed, covering staged pickup and retry, local presence, the relay mask, and the legacy fast path. The full command encountered a sandbox denial binding to `127.0.0.1` in the existing HTTP fixtures.
- Remaining work is Part 2 STAND, then final verification.

## Part 2 complete

- Wide parcels enter STAND before PICK. TableGrid projects the detected parcel box onto the measured mat when available. The fallback uses a short side greater than 130 px at 1280 × 720. Both recorded parcel fixtures tested wide.
- STAND hides every bin and uses the destination lane's arm to lift while the opposite arm steadies the parcel. Two consecutive observations at least 40% below the original footprint trigger the top-grasp PICK. Two attempts are allowed by default with a configurable call limit. Exhausted STAND attempts end the delivery through HOME.
- Eighteen targeted tests passed. Full-suite verification is limited by the sandbox's localhost restriction. The staging flag defaults to false. Hardware calibration and validation are outside this task.

### Verification

- The focused state machine, table grid, real-profile, and overlay suites passed 62 tests with the four localhost-dependent cases deselected.
- The full `.venv/bin/python -m pytest -q` run reported 185 passed and 2 skipped. The sandbox's localhost bind restriction caused 10 failures and 27 setup errors.
- `git diff --check` passed. Changes are uncommitted on `main`.
- The final offline-only run passed 179 tests with 2 skipped. Seven additional unit tests from the excluded integration modules passed separately.
