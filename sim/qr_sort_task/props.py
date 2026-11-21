"""The real props, measured by hand: one place for make_usd.py (geometry, colours) and build_task.py (layout, success check).

ACTIVE = the seven parcels the user actually has (the only ones in build_task CASES / stream); the other PARCELS stay defined, unused.
Receptacles: two open shoe boxes (lids off) for the lanes and a LOW reject tray (the black shoe box's lid, upside down) to swipe rejects into. Size = outer length x width x height, measured in inches.
Length runs away from the robot (env x), width runs across the table (env y). Colours are sRGB medians sampled from the photos
(room light, phone camera); calibrate the harness HSV ranges on sim/real head-camera frames, not on these numbers.
"""
import hashlib

IN = 0.0254
WALL = 0.005  # modelled wall/floor thickness (m): real board is ~2-3 mm, 5 mm keeps a falling parcel from tunnelling through

GRIPPER_OPEN = 0.0975  # G gripper fully open between the fingers (2 x 48.75 mm, robot URDF): a parcel's grasp width must be under it
QR_MM, MARGIN_MM = 40.0, 5.0  # QR symbol width and the MINIMUM white quiet zone round it (the label art uses max(this, 4 modules))
LABEL_SIZES = {"large": (100, 70), "small": (62, 62)}  # shipping label sizes (mm), drawn by sim/make_qr_boxes.label_image
CUBE = (0.06, 0.06, 0.06)  # sim stand-in for a parcel that has no measured prop yet
FACES = ("pz", "nz", "py", "ny", "px", "nx")  # box faces +z top, -z bottom, +-y sides, +-x ends (nx faces the robot)

# Measured parcels, keyed by the parcel id build_task.py uses (z95112 = lane A, QR 420/95112/001; z90210 = lane B; z10001 = unknown
# ZIP -> reject). size = the parcel as it ARRIVES in the induction zone: x away from the robot, y across the table, z up (fingers close
# sideways, so y is the grasp width). yaw = how far (rad) the human may leave it turned; mass = spawn range (kg, guessed: weigh them).
# color = body sRGB from the photos. tape = a stripe of tape / print looping round the parcel: centred `at` mm from the middle along
# `axis`, `mm` wide along it, on the four faces parallel to that axis. A parcel with a QR code (ROUTES) carries its ONE shipping label (`label` = size key
# in LABEL_SIZES, centred) on a RANDOM face each episode: variants() builds one variant per face that can hold it, each with labels=(face,); a prop
# without a code keeps labels=().
# patches = printed rectangles: face, centre `at` (u, v) mm from the face centre in its tile axes, size `mm` (u, v), drawn in order.
# finish = OmniPBR metallic / roughness for a shiny parcel (default matte paper). Not listed = still the 60 mm sim cube (red 50 mm cube
# for garbage).
PARCELS = {
    # package 1: Amazon box 9 x 6 x 3.5 in, stood on its long edge, long side pointing away from the robot: the only pose the gripper
    # can close on (3.5 in = 89 mm across). Seam tape = belt round it at mid height.
    "z95112": dict(label="large", size=(9 * IN, 3.5 * IN, 6 * IN), mass=(0.10, 0.20), yaw=0.1, color=(204, 166, 118),
                   tape=dict(axis="z", mm=72, at=0, rgb=(33, 30, 28))),
    # package 2: Amazon box 11 x 8.1 x 4 in, lying flat, long side ACROSS the table (lengthwise it does not fit in front of the bins).
    # NOT graspable by the G gripper: thinnest side 4 in = 102 mm. Seam tape along its length over the top and down both ends.
    "z90210": dict(label="large", size=(8.1 * IN, 11 * IN, 4 * IN), mass=(0.15, 0.30), yaw=0.1, color=(188, 157, 123),
                   tape=dict(axis="x", mm=72, at=0, rgb=(40, 40, 40))),
    # package 3 = the "mailer" (soft plastic bag parcel): Amazon padded mailer 10 x 8.5 x 0.5 in, lying flat, long side across. NOT graspable
    # (13 mm thick, 216-254 mm wide), rigid in sim (the real one bends). Off-white; blue tear band across it 72 mm from the middle.
    # Flat: the label (LL-03, unknown ZIP -> reject) goes on the top or bottom face only.
    "z10001": dict(label="large", size=(8.5 * IN, 10 * IN, 0.5 * IN), mass=(0.05, 0.10), yaw=0.1, color=(231, 222, 205), faces=("pz", "nz"),
                   tape=dict(axis="y", mm=60, at=72, rgb=(49, 100, 179))),
    # package 4 = the "perfume" item (a retail box with no shipping label -> reject tray): perfume box 6 x 4 x 2 in, brushed light-blue foil,
    # lying FLAT, biggest face down (like every active parcel; 102 mm wide: not graspable by one gripper). Grey-bordered white name label on
    # the top. The real box also has a small authenticity sticker with its own QR; not modelled. Mass guessed for a boxed 100 ml bottle.
    "garbage": dict(size=(6 * IN, 4 * IN, 2 * IN), mass=(0.25, 0.40), yaw=0.3, color=(109, 128, 145), labels=(),
                    finish=dict(metallic=0.5, roughness=0.35),
                    patches=[dict(face="pz", at=(-30, 12), mm=(58, 58), rgb=(175, 178, 182)),
                             dict(face="pz", at=(-30, 12), mm=(40, 28), rgb=(246, 240, 224))]),
    # --- sim-only props (not measured or photographed yet): sizes are nominal, colours and masses guessed ---
    # small kraft box 3 x 3 x 3 in: a cube, graspable on any side (76 mm). Lane A (ZIP3 941).
    "z94110": dict(label="small", size=(3 * IN, 3 * IN, 3 * IN), mass=(0.06, 0.12), yaw=0.2, color=(190, 150, 102)),
    # tall kraft box 3 x 5 x 8 in, arrives standing on its 5 x 3 in end, 3 in = 76 mm across (graspable). Lane B (ZIP3 900). 8 in = 203 mm
    # is taller than the bin walls, so it stays proud of the red bin. Black tape belt at mid height.
    "z90015": dict(label="large", size=(5 * IN, 3 * IN, 8 * IN), mass=(0.15, 0.30), yaw=0.2, color=(178, 140, 96),
                   tape=dict(axis="z", mm=48, at=0, rgb=(33, 30, 28))),
    # white retail-style box 5 x 3.5 x 2.5 in, stood on its 5 x 2.5 in edge: 63.5 mm across (~17 mm spare per finger). Unknown ZIP -> reject.
    "z60601": dict(label="small", size=(5 * IN, 2.5 * IN, 3.5 * IN), mass=(0.10, 0.20), yaw=0.2, color=(236, 234, 228)),
    # large kraft box 12 x 10 x 6 in, flat, long side across the table. NOT graspable by one gripper (narrowest side from above 254 mm,
    # 152 mm even on edge): parcel_report WARNs, intended. Lane A (ZIP3 950). yaw is small: at 0.1 rad it no longer fits between the
    # bins and the table edge. Tape belt round it.
    "z95014": dict(label="large", size=(10 * IN, 12 * IN, 6 * IN), mass=(0.40, 0.80), yaw=0.02, color=(176, 140, 98),
                   tape=dict(axis="z", mm=72, at=0, rgb=(40, 40, 40))),
    # brown leather wallet 4.5 x 3.5 x 0.75 in, flat, no label, matte dark brown. Not a parcel -> reject. 89 mm across: tight grasp.
    "wallet": dict(size=(4.5 * IN, 3.5 * IN, 0.75 * IN), mass=(0.08, 0.15), yaw=0.3, color=(60, 40, 28), labels=()),
    # --- real FLAT retail boxes (sim-modelled from phone measurements, photos not in the repo yet). They always arrive lying flat, biggest
    # face down; `faces` restricts the one label to the big top/bottom face (variants). All three are wider than the gripper opening
    # lying flat (parcel_report WARNs, intended): they are handled with two arms. ---
    # Bhindi Masala ready-meal box 6.5 x 4 x 1.5 in: deep purple, orange band at one end, dark food photo on top. Lane A (spare LL-08, 95126).
    "bhindi": dict(label="large", size=(6.5 * IN, 4 * IN, 1.5 * IN), mass=(0.15, 0.30), yaw=0.1, color=(70, 40, 110), faces=("pz", "nz"),
                   tape=dict(axis="x", mm=30, at=67, rgb=(225, 120, 40)),
                   patches=[dict(face="pz", at=(-20, 0), mm=(90, 80), rgb=(35, 22, 22))]),
    # IQBAR (stylised brand name) protein bar variety pack 10.5 x 4 x 2.5 in: chocolate brown, dark band along its length. yaw is small: 267 mm
    # lengthwise barely fits in front of the bins: yaw 0, the centre has only ~6 mm of x play. Lane B (spare LL-11, 90024).
    "iqbar_a": dict(label="large", size=(10.5 * IN, 4 * IN, 2.5 * IN), mass=(0.30, 0.55), yaw=0.0, color=(95, 60, 40), faces=("pz", "nz"),
                  tape=dict(axis="x", mm=40, at=0, rgb=(40, 30, 25))),
    # a second IQBAR box: identical geometry and print, NO label -> reject. (Same body as iqbar_a minus its label.)
    "iqbar_b": dict(size=(10.5 * IN, 4 * IN, 2.5 * IN), mass=(0.30, 0.55), yaw=0.0, color=(95, 60, 40), labels=(),
                    tape=dict(axis="x", mm=40, at=0, rgb=(40, 30, 25))),
    # Miro box 8.5 x 6.5 x 1.5 in: black, white logo patch on top (plain rectangle; the real one is rounded). Reject (spare LL-14, 02139).
    # The logo sits 55 mm off-centre so the centred shipping label does not hide it.
    "miro": dict(label="large", size=(8.5 * IN, 6.5 * IN, 1.5 * IN), mass=(0.20, 0.40), yaw=0.1, color=(25, 25, 28), faces=("pz", "nz"),
                 patches=[dict(face="pz", at=(0, 55), mm=(90, 45), rgb=(235, 235, 235))]),
    # plain kraft box 6 x 4 x 3 in, no label (a random box), stood on its 6 x 3 in edge: 76 mm across. -> reject.
    "plainbox": dict(size=(6 * IN, 3 * IN, 4 * IN), mass=(0.10, 0.20), yaw=0.2, color=(196, 158, 110), labels=()),
}

ACTIVE = ("bhindi", "iqbar_a", "miro", "z10001", "iqbar_b", "garbage", "wallet")  # the parcels the user has: bhindi/iqbar_a -> blue/red lane, rest reject

SORT_PLAN = {"950": "bin_blue", "951": "bin_blue", "940": "bin_blue", "941": "bin_blue",
             "900": "bin_red", "901": "bin_red", "902": "bin_red"}  # ZIP3 -> lane bin (harness config sort plan); anything else -> reject_bin
LANE = {"bin_blue": "A", "bin_red": "B", "reject_bin": "reject"}
SHIP_DATE, SENDER = "2026-11-20", ("LabelLoop Fulfillment", "1 Washington Sq", "San Jose CA 95192")  # fictional, same on every label

# Label registry: ONE row per printed label (sim parcel id or "" for a spare, real item, fictional recipient). Payload 420/<ZIP5>/<seq3>
# (GS1 AI 420 = ship-to ZIP) is what the robot routes on; everything else is decoration drawn by make_qr_boxes.label_image.
_ROWS = [
    ("z95112", "package 1: Amazon box 9 x 6 x 3.5 in", "Maya Okafor", "212 Maple Ct", "San Jose", "CA", "95112"),
    ("z90210", "package 2: Amazon box 11 x 8.1 x 4 in", "Daniel Reyes", "9 Canon Dr", "Beverly Hills", "CA", "90210"),
    ("z10001", "package 3: Amazon padded mailer 10 x 8.5 in (mailer, soft plastic bag parcel)", "Priya Natarajan", "350 W 31st St", "New York", "NY", "10001"),
    ("z94110", "sim only (small kraft box 3 x 3 x 3 in), not yet bought", "Ethan Brooks", "88 Valencia Ave", "San Francisco", "CA", "94110"),
    ("z90015", "sim only (tall kraft box 3 x 5 x 8 in), not yet bought", "Sofia Lindqvist", "1400 S Figueroa St", "Los Angeles", "CA", "90015"),
    ("z60601", "sim only (white retail box 5 x 3.5 x 2.5 in), not yet bought", "Marcus Webb", "77 E Randolph St", "Chicago", "IL", "60601"),
    ("z95014", "sim only (large kraft box 12 x 10 x 6 in), not yet bought", "Hana Kimura", "10300 De Anza Blvd", "Cupertino", "CA", "95014"),
    ("bhindi", "Bhindi Masala ready-meal box 6.5 x 4 x 1.5 in", "Olivia Grant", "415 Union Ave", "San Jose", "CA", "95126"),
    ("", "spare, lane A", "Noah Patel", "600 Castro St", "Mountain View", "CA", "94043"),
    ("", "spare, lane A", "Lena Fischer", "2250 Lawrence Rd", "Santa Clara", "CA", "95054"),
    ("iqbar_a", "IQBAR protein bar pack 10.5 x 4 x 2.5 in", "Carlos Mendoza", "405 Hilgard Ave", "Los Angeles", "CA", "90024"),
    ("", "spare, lane B", "Ruth Adler", "200 N Spring St", "Los Angeles", "CA", "90012"),
    ("", "spare, lane B", "Tomas Novak", "1601 Abbot Kinney Blvd", "Venice", "CA", "90291"),
    ("miro", "Miro box 8.5 x 6.5 x 1.5 in", "Aiko Tanaka", "33 Oxford St", "Cambridge", "MA", "02139"),
    ("", "spare, reject", "Grace Liu", "1100 Congress Ave", "Austin", "TX", "73301"),
    ("", "spare, reject", "Owen Doyle", "400 Pine St", "Seattle", "WA", "98101"),
]


def _label(i, sim, item, name, street, city, state, zip5):
    lid = f"LL-{i + 1:02d}"
    h = int(hashlib.md5(lid.encode()).hexdigest(), 16)
    d = f"{h % 10 ** 14:014d}"
    tgt = SORT_PLAN.get(zip5[:3], "reject_bin")
    return dict(label_id=lid, qr_payload=f"420/{zip5}/001", zip5=zip5, zip3=zip5[:3], lane=LANE[tgt], target_bin=tgt, recipient=name,
                street=street, city=city, state=state, tracking=f"LLX {d[:4]} {d[4:8]} {d[8:12]} {d[12:]}",
                weight_lb=f"{0.5 + h // 7 % 40 / 10:.1f}", date=SHIP_DATE, size=PARCELS[sim]["label"] if sim else "large", sim_id=sim, item=item)


LABELS = [_label(i, *r) for i, r in enumerate(_ROWS)]
ROUTES = {r["sim_id"]: r["qr_payload"] for r in LABELS if r["sim_id"]}  # labelled parcel id -> QR payload
LABEL_OF = {r["sim_id"]: r for r in LABELS if r["sim_id"]}


def fits(face_m, size):
    """True if a label of LABEL_SIZES[size] fits a face of face_m (m x m) in either orientation."""
    (a, b), (s, l) = sorted(face_m), sorted(v / 1000 for v in LABEL_SIZES[size])
    return a >= s - 1e-9 and b >= l - 1e-9


def variants(base):
    """{variant id: spec} of parcel `base`. A parcel with a QR code (ROUTES) gets one variant `<base>_<face>` per face that can hold its
    label (fits) and is in the prop's optional `faces` (flat boxes: pz/nz only), labels=(face,); any other prop is {base: its spec} (None for the plain cube)."""
    p = PARCELS.get(base)
    if base not in ROUTES:
        return {base: p}
    x, y, z = p["size"]
    dims = {"pz": (x, y), "nz": (x, y), "py": (x, z), "ny": (x, z), "px": (y, z), "nx": (y, z)}
    return {f"{base}_{f}": dict(p, labels=(f,)) for f in p.get("faces", FACES) if fits(dims[f], p["label"])}

# Table mat: the real event benches have a mint/teal green work mat over the whole tabletop (WALLX was likely trained on those).
# size (x depth, y across, z thickness) m, sRGB matte; colour sampled from the event photo, adjust if needed.
TABLE_MAT = dict(size=(0.675, 1.10, 0.003), color=(46, 178, 150))  # 5 mm inset per side: the layout solver rejects a mat as big as the tabletop

BINS = {  # asset/scene id -> outer size (m) and sRGB colours
    "bin_blue": dict(size=(14 * IN, 8 * IN, 5 * IN), outside=(40, 57, 73), inside=(223, 178, 112)),       # navy shoe box, kraft inside: lane A
    "bin_red": dict(size=(14 * IN, 9.5 * IN, 5 * IN), outside=(235, 30, 30), inside=(236, 188, 124)),     # red shoe box, kraft inside: lane B
    # reject TRAY (id kept so harness/config keep working): the black shoe box's LID upside down, rim 1.0 in high, so a parcel can be swiped
    # over the low rim into it. Black outside, white inside. NOT a bin: success = parcel centre inside the footprint and resting.
    "reject_bin": dict(size=(13.2 * IN, 9.2 * IN, 1.0 * IN), outside=(40, 40, 38), inside=(245, 240, 225)),
}
