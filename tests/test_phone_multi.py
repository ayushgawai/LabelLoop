"""Case 2b: phone scanner with several parcels (scenarios/wx/case2_multi*.json).

Each phone read waits for its own new scan (never an earlier parcel's), autopick takes the parcel the phone read (head-camera
label match, else nearest the phone's aim, else largest), an "if_parcel" step ends the run when the table is clear, and the phone
decode puts the label nearest the frame centre first."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from labelloop import sequence
from labelloop.dashboard import package
from labelloop.dashboard.track import find_parcel, find_parcels
from labelloop.sequence import Sequence, phone_code

REPO = Path(__file__).resolve().parents[1]
A, B = "420/94043/001", "420/90210/001"  # zone 1 (blue) and zone 2 (red)
BOX_A, BOX_B = (300, 400, 140, 100), (700, 560, 80, 60)  # large parcel left of centre, small one right


def table(*boxes):
    """1280x720 head frame: green mat everywhere, dark parcels at boxes (x, y, w, h). Returns RGB."""
    img = np.zeros((720, 1280, 3), np.uint8)
    img[:] = (90, 140, 40)  # BGR mat (what find_parcel_px paints outside its roi)
    for x, y, w, h in boxes:
        img[y:y + h, x:x + w] = (40, 60, 90)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def centre(b):
    return (b[0] + b[2] / 2, b[1] + b[3] / 2)


def write_board(path, code, t):
    path.write_text(json.dumps({"current": {"code": code, "scanned": t, "last_seen": t}}))


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def clock(monkeypatch):
    import time
    c = Clock()
    monkeypatch.setattr(time, "time", c)
    return c


@pytest.fixture
def board(tmp_path, monkeypatch):
    p = tmp_path / "board.json"
    monkeypatch.setattr(sequence, "BOARD", p)
    return p


def state(pos=(0.0, 0.0, 0.2)):
    ee = lambda: {"position": list(pos), "rotation_rpy": [0.0, 0.0, 0.0], "gripper": 0.5}  # noqa: E731
    return {"left_ee": ee(), "right_ee": ee()}


# --- parcels on the head camera --------------------------------------------------------------------------------------------------

def test_find_parcels_lists_every_parcel_largest_first():
    img = cv2.cvtColor(table(BOX_B, BOX_A), cv2.COLOR_RGB2BGR)
    boxes = find_parcels(img)
    assert boxes == [BOX_A, BOX_B]
    assert find_parcel(img) == BOX_A  # unchanged: the largest
    assert find_parcels(cv2.cvtColor(table(), cv2.COLOR_RGB2BGR)) == [] and find_parcel(cv2.cvtColor(table(), cv2.COLOR_RGB2BGR)) is None


@pytest.fixture
def no_reads(monkeypatch):
    monkeypatch.setattr(sequence, "locate", lambda img, *a, **k: [])
    monkeypatch.setattr(sequence, "any_barcodes", lambda img: [])
    monkeypatch.setattr(sequence.label_ocr, "zips", lambda img, roi, *a, **k: [])


def test_scanned_parcel_is_the_one_with_the_scanned_label(monkeypatch, no_reads):
    monkeypatch.setattr(sequence, "locate", lambda img, *a, **k: [(B, centre(BOX_B)), (A, centre(BOX_A))])
    sq = Sequence({"steps": []})
    sq.code = B
    assert sq._scanned_parcel(table(BOX_A, BOX_B), {"target": "scan"}) == BOX_B
    assert sq.pick_match == "label"
    sq.code = A
    assert sq._scanned_parcel(table(BOX_A, BOX_B), {"target": "scan"}) == BOX_A


def test_scanned_parcel_pincode_ocr_label_counts(monkeypatch, no_reads):
    monkeypatch.setattr(sequence.label_ocr, "zips", lambda img, roi, *a, **k: [("90210", centre(BOX_B))])
    sq = Sequence({"steps": []})
    sq.code = "420/90210/000"  # a bare pincode from the phone
    assert sq._scanned_parcel(table(BOX_A, BOX_B), {"target": "scan"}) == BOX_B and sq.pick_match == "label"


def test_scanned_parcel_falls_back_to_the_aim_then_the_largest(no_reads):
    sq = Sequence({"steps": []})
    sq.code = B
    aim = [centre(BOX_B)[0] / 1280, centre(BOX_B)[1] / 720]
    assert sq._scanned_parcel(table(BOX_A, BOX_B), {"target": "scan", "aim": aim}) == BOX_B and sq.pick_match == "aim"
    assert sq._scanned_parcel(table(BOX_A, BOX_B), {"target": "scan"}) == BOX_A and sq.pick_match == "largest"
    assert sq._scanned_parcel(table(BOX_A, BOX_B), {"target": "scan", "roi": [0.5, 1.0]}) == BOX_B and sq.pick_match == "only"
    assert sq._scanned_parcel(table(), {"target": "scan"}) is None


def test_a_label_far_from_every_parcel_is_no_match(monkeypatch, no_reads):
    monkeypatch.setattr(sequence, "locate", lambda img, *a, **k: [(B, (1200.0, 100.0))])  # e.g. the label of a parcel in a bin
    sq = Sequence({"steps": []})
    sq.code = B
    assert sq._scanned_parcel(table(BOX_A, BOX_B), {"target": "scan"}) == BOX_A and sq.pick_match == "largest"


# --- the phone board -------------------------------------------------------------------------------------------------------------

def test_phone_code_since_and_exclude(board, clock):
    write_board(board, A, 999.0)
    assert phone_code(30) == A
    assert phone_code(30, since=999.5) is None  # made before this read step started
    assert phone_code(30, exclude=[A]) is None  # an earlier parcel's code
    assert phone_code(30, exclude=[B]) == A
    clock.t = 1100.0
    assert phone_code(30) is None  # too old


def run_reads(board, clock, spec, script):
    """Step a Sequence; script = {call index: (code, time) to put on the phone board before that call}."""
    sq, img, out = Sequence(spec), {"front": table()}, []
    for k in range(60):
        if k in script:
            write_board(board, *script[k])
        clock.t += 1.0
        instr = sq.step(img, state())
        out.append((sq.i, sq.code, instr))
        if sq.done:
            break
    return sq, out


TWO_READS = [{"read": True, "from_phone": True, "max_age": 300, "wait_calls": 20, "required": True},
             {"move": {"left": {"position": [0, 0, 0.03]}}, "calls": 3},
             {"read": True, "from_phone": True, "max_age": 300, "wait_calls": 20, "required": True},
             {"move": {"left": {"position": [0, 0, 0.03]}}, "calls": 3},
             {"move": {"left": {"position": [0, 0, 0]}}, "calls": 2}]


def test_second_phone_read_waits_for_a_scan_made_after_it_started(board, clock):
    # parcel 1 scanned at call 0; parcel 2's read starts around call 3 and must NOT take parcel 1's code: it waits, then gets B
    sq, out = run_reads(board, clock, {"steps": TWO_READS}, {0: (A, 1000.5), 10: (B, 1012.0)})
    assert out[0][1] == A
    waiting = [o for o in out if o[0] == 2 and o[2] is None]
    assert len(waiting) >= 5 and all(o[1] == A for o in waiting)  # held still: parcel 1's zone is not reused for parcel 2
    assert sq.code == B and sq.phone_codes == [A, B] and sq.done


def test_new_code_ignores_a_parcel_already_sorted(board, clock):
    steps = [dict(s, new_code=True) if s.get("from_phone") else s for s in TWO_READS]
    # the phone sees parcel 1 again (in its bin) right when read 2 starts: ignored; then parcel 2
    sq, out = run_reads(board, clock, {"steps": steps}, {0: (A, 1000.5), 4: (A, 1004.5), 5: (A, 1005.5), 9: (B, 1010.5)})
    assert sq.code == B and sq.phone_codes == [A, B]
    # without new_code the re-seen parcel 1 would have been taken for parcel 2
    sq2, _ = run_reads(board, clock, {"steps": TWO_READS}, {0: (A, clock.t + 0.5), 4: (A, clock.t + 4.5)})
    assert sq2.phone_codes == [A, A]


def test_if_parcel_goes_home_when_the_table_is_clear(monkeypatch):
    steps = [{"if_parcel": True, "roi": [0.38, 1.0]}, {"move": {"right": {"position": [0, 0, 0.1]}}, "calls": 2},
             {"move": {"left": {"position": [0, 0, 0]}}, "calls": 2}]
    monkeypatch.setattr(sequence, "find_parcel_px", lambda img, roi=None: None)
    sq = Sequence({"steps": steps})
    assert sq.step({"front": table()}, state()) == "(scripted move 2)"
    assert "no parcel left on the table: all sorted, going home" in sq.last_log["flags"]
    monkeypatch.setattr(sequence, "find_parcel_px", lambda img, roi=None: (640.0, 400.0))
    sq = Sequence({"steps": steps})
    assert sq.step({"front": table()}, state()) == "(scripted move 1)"
    sq = Sequence({"steps": [{"if_parcel": True}]})  # as the last step: finishes, no loop
    monkeypatch.setattr(sequence, "find_parcel_px", lambda img, roi=None: None)
    assert sq.step({"front": table()}, state()) is None and sq.done


# --- phone frames with several labels --------------------------------------------------------------------------------------------

def qr(text, size=220):
    m = cv2.QRCodeEncoder.create().encode(text)
    return cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)


def frame_with(centre_code, corner_code):
    img = np.full((1080, 1440), 255, np.uint8)
    img[430:650, 610:830] = qr(centre_code)  # where the phone points
    img[40:260, 40:260] = qr(corner_code)
    ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return jpg.tobytes()


def test_phone_decode_puts_the_centre_label_first():
    assert package.decode_jpeg(frame_with(B, A)) == [B, A]
    assert package.decode_jpeg(frame_with(A, B)) == [A, B]


# --- the recipe ------------------------------------------------------------------------------------------------------------------

def test_case2_multi_recipe_and_dashboard_story_line_up():
    rec = json.loads((REPO / "scenarios/wx/case2_multi.json").read_text(encoding="utf-8"))
    steps = json.loads((REPO / "scenarios/wx/case2_multi_steps.json").read_text(encoding="utf-8"))["steps"]
    assert "case2_multi_steps.json" in rec["relay"] and rec["seconds"] <= 480  # one 8-minute case
    reads = [s for s in steps if s.get("from_phone")]
    picks = [s["autopick"] for s in steps if "autopick" in s]
    assert len(reads) == len(picks) == 3 and all(r["new_code"] and r["required"] for r in reads)
    assert all(p["target"] == "scan" and p["arm"] == "right" for p in picks)
    assert sum("if_parcel" in s for s in steps) == 2 and "move" in steps[-1] and "left" in steps[-1]["move"]
    demo = next(d for d in json.loads((REPO / "labelloop/dashboard/demos.json").read_text(encoding="utf-8")) if d["id"] == "case2m")
    assert demo["run"] == "scenarios/wx/case2_multi.json" and demo["relay"] == rec["relay"] and not demo.get("hidden")
    assert len(demo["steps"]) == len(steps) + 1 and demo["steps"][-1]["who"] == "done"
    for d, s in zip(demo["steps"], steps):
        assert (d["who"] == "wallx") == ("say" in s)


def test_case2_multi_dry_run_sorts_two_parcels_then_goes_home(board, clock, monkeypatch):
    """The whole recipe with fakes: the phone scans A then B, each is picked and 'dropped' at its zone's drop point, then the
    head camera shows an empty table and the left arm goes home with the phone."""
    monkeypatch.chdir(REPO)  # replay paths are repo-relative
    spec = json.loads((REPO / "scenarios/wx/case2_multi_steps.json").read_text(encoding="utf-8"))
    left = {"n": 2}  # parcels still on the table
    monkeypatch.setattr(sequence, "find_parcel_px", lambda img, roi=None: (800.0, 450.0) if left["n"] else None)
    monkeypatch.setattr(Sequence, "_scanned_parcel", lambda self, front, c: (760, 420, 80, 60) if left["n"] else None)
    monkeypatch.setattr(Sequence, "_pick_spot_covered", lambda self, front: False)
    sq, img, sorted_, said = Sequence(spec), {"front": table()}, [], []
    write_board(board, B, clock.t - 50)  # an old scan from before the run never counts
    for k in range(3000):
        clock.t += 0.5
        st = state()
        s = sq.step_spec
        if s is not None and "say" in s:  # WALLX "carries" it: the right arm is at the zone's drop point, lowered
            st["right_ee"]["position"] = sequence.DROP_POINTS[sq.mat]["right"] + [0.05]
        if s is not None and s.get("from_phone"):  # the phone over the table: parcel A first, then B
            write_board(board, B if sq.phone_codes else A, clock.t + 0.1)
        before = sq.i
        instr = sq.step(img, st)
        if instr and not instr.startswith("("):
            said.append(instr)
        if "release" in (spec["steps"][before] if before < len(spec["steps"]) else {}) and sq.i != before:
            sorted_.append((sq.code, sq.mat))
            left["n"] -= 1
        if sq.done:
            break
    assert sq.done and sq.phone_codes == [A, B]
    assert sorted_ == [(A, "blue"), (B, "red")]
    zones = spec["zones"]
    assert said[0] == "Stretch and put the package you are holding into zone %s." % zones["blue"]
    assert said[-1] == "Stretch and put the package you are holding into zone %s." % zones["red"]
    assert sq.i >= len(spec["steps"]) - 1  # round 3 skipped: straight to the last step (left arm home with the phone)
