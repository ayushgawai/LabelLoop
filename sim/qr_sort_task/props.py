"""The real props, measured by hand: one place for make_usd.py (geometry, colours) and build_task.py (layout, success check).

Bins are the three open shoe boxes (lids off) the robot sorts into. Size = outer length x width x height, measured in inches.
Length runs away from the robot (env x), width runs across the table (env y). Colours are sRGB medians sampled from the photos
(room light, phone camera); calibrate the harness HSV ranges on sim/real head-camera frames, not on these numbers.
"""
IN = 0.0254
WALL = 0.005  # modelled wall/floor thickness (m): real board is ~2-3 mm, 5 mm keeps a falling parcel from tunnelling through

GRIPPER_OPEN = 0.0975  # G gripper fully open between the fingers (2 x 48.75 mm, robot URDF): a parcel's grasp width must be under it
QR_MM, MARGIN_MM = 40.0, 5.0  # QR label = symbol + white quiet zone on every side (50 mm), printed by sim/make_qr_boxes.py
LABEL = (QR_MM + 2 * MARGIN_MM) / 1000
CUBE = (0.06, 0.06, 0.06)  # sim stand-in for a parcel that has no measured prop yet
FACES = ("pz", "nz", "py", "ny", "px", "nx")  # box faces +z top, -z bottom, +-y sides, +-x ends (nx faces the robot)

# Measured parcels, keyed by the parcel id build_task.py uses (z95112 = lane A, QR 420/95112/001; z90210 = lane B; z10001 = unknown
# ZIP -> reject). size = the parcel as it ARRIVES in the induction zone: x away from the robot, y across the table, z up (fingers close
# sideways, so y is the grasp width). yaw = how far (rad) the human may leave it turned; mass = spawn range (kg, guessed: weigh them).
# color = body sRGB from the photos. tape = a stripe of tape / print looping round the parcel: centred `at` mm from the middle along
# `axis`, `mm` wide along it, on the four faces parallel to that axis. A parcel with a QR code (ROUTES) carries its ONE 50 mm label
# (centred) on a RANDOM face each episode: variants() builds one variant per face that can hold it, each with labels=(face,); a prop
# without a code keeps labels=().
# patches = printed rectangles: face, centre `at` (u, v) mm from the face centre in its tile axes, size `mm` (u, v), drawn in order.
# finish = OmniPBR metallic / roughness for a shiny parcel (default matte paper). Not listed = still the 60 mm sim cube (red 50 mm cube
# for garbage).
PARCELS = {
    # package 1: Amazon box 9 x 6 x 3.5 in, stood on its long edge, long side pointing away from the robot: the only pose the gripper
    # can close on (3.5 in = 89 mm across). Seam tape = belt round it at mid height.
    "z95112": dict(size=(9 * IN, 3.5 * IN, 6 * IN), mass=(0.10, 0.20), yaw=0.1, color=(204, 166, 118),
                   tape=dict(axis="z", mm=72, at=0, rgb=(33, 30, 28))),
    # package 2: Amazon box 11 x 8.1 x 4 in, lying flat, long side ACROSS the table (lengthwise it does not fit in front of the bins).
    # NOT graspable by the G gripper: thinnest side 4 in = 102 mm. Seam tape along its length over the top and down both ends.
    "z90210": dict(size=(8.1 * IN, 11 * IN, 4 * IN), mass=(0.15, 0.30), yaw=0.1, color=(188, 157, 123),
                   tape=dict(axis="x", mm=72, at=0, rgb=(40, 40, 40))),
    # package 3: Amazon padded mailer 10 x 8.5 x 0.5 in, lying flat, long side across. NOT graspable (13 mm thick, 216-254 mm wide),
    # and rigid in sim (the real one bends). Off-white; blue tear band across it 72 mm from the middle. 
    "z10001": dict(size=(8.5 * IN, 10 * IN, 0.5 * IN), mass=(0.05, 0.10), yaw=0.1, color=(231, 222, 205),
                   tape=dict(axis="y", mm=60, at=72, rgb=(49, 100, 179))),
    # package 4 = the garbage item (a retail box with no shipping label -> black bin): perfume box 6 x 4 x 2 in, brushed light-blue foil,
    # stood on its long edge like package 1 (2 in = 51 mm across, ~23 mm spare per finger). Grey-bordered white name label on one big
    # side. The real box also has a small authenticity sticker with its own QR; not modelled. Mass guessed for a boxed 100 ml bottle.
    "garbage": dict(size=(6 * IN, 2 * IN, 4 * IN), mass=(0.25, 0.40), yaw=0.3, color=(109, 128, 145), labels=(),
                    finish=dict(metallic=0.5, roughness=0.35),
                    patches=[dict(face="py", at=(-30, 12), mm=(58, 58), rgb=(175, 178, 182)),
                             dict(face="py", at=(-30, 12), mm=(40, 28), rgb=(246, 240, 224))]),
    # --- sim-only props (not measured or photographed yet): sizes are nominal, colours and masses guessed ---
    # small kraft box 3 x 3 x 3 in: a cube, graspable on any side (76 mm). Lane A (ZIP3 941).
    "z94110": dict(size=(3 * IN, 3 * IN, 3 * IN), mass=(0.06, 0.12), yaw=0.2, color=(190, 150, 102)),
    # tall kraft box 3 x 5 x 8 in, arrives standing on its 5 x 3 in end, 3 in = 76 mm across (graspable). Lane B (ZIP3 900). 8 in = 203 mm
    # is taller than the bin walls, so it stays proud of the red bin. Black tape belt at mid height.
    "z90015": dict(size=(5 * IN, 3 * IN, 8 * IN), mass=(0.15, 0.30), yaw=0.2, color=(178, 140, 96),
                   tape=dict(axis="z", mm=48, at=0, rgb=(33, 30, 28))),
    # white retail-style box 5 x 3.5 x 2.5 in, stood on its 5 x 2.5 in edge: 63.5 mm across (~17 mm spare per finger). Unknown ZIP -> reject.
    "z60601": dict(size=(5 * IN, 2.5 * IN, 3.5 * IN), mass=(0.10, 0.20), yaw=0.2, color=(236, 234, 228)),
    # large kraft box 12 x 10 x 6 in, flat, long side across the table. NOT graspable by one gripper (narrowest side from above 254 mm,
    # 152 mm even on edge): parcel_report WARNs, intended. Lane A (ZIP3 950). yaw is small: at 0.1 rad it no longer fits between the
    # bins and the table edge. Tape belt round it.
    "z95014": dict(size=(10 * IN, 12 * IN, 6 * IN), mass=(0.40, 0.80), yaw=0.02, color=(176, 140, 98),
                   tape=dict(axis="z", mm=72, at=0, rgb=(40, 40, 40))),
    # brown leather wallet 4.5 x 3.5 x 0.75 in, flat, no label, matte dark brown. Not a parcel -> reject. 89 mm across: tight grasp.
    "wallet": dict(size=(4.5 * IN, 3.5 * IN, 0.75 * IN), mass=(0.08, 0.15), yaw=0.3, color=(60, 40, 28), labels=()),
    # plain kraft box 6 x 4 x 3 in, no label (a random box), stood on its 6 x 3 in edge: 76 mm across. -> reject.
    "plainbox": dict(size=(6 * IN, 3 * IN, 4 * IN), mass=(0.10, 0.20), yaw=0.2, color=(196, 158, 110), labels=()),
}

ROUTES = {"z95112": "420/95112/001", "z90210": "420/90210/001", "z10001": "420/10001/001", "z94110": "420/94110/001",
          "z90015": "420/90015/001", "z60601": "420/60601/001", "z95014": "420/95014/001"}  # labelled parcel id -> QR payload


def variants(base):
    """{variant id: spec} of parcel `base`. A parcel with a QR code (ROUTES) gets one variant `<base>_<face>` per face that can hold the
    label (both face dimensions >= LABEL), labels=(face,); any other prop is {base: its spec} (None for the plain cube)."""
    p = PARCELS.get(base)
    if base not in ROUTES:
        return {base: p}
    x, y, z = p["size"]
    dims = {"pz": (x, y), "nz": (x, y), "py": (x, z), "ny": (x, z), "px": (y, z), "nx": (y, z)}
    return {f"{base}_{f}": dict(p, labels=(f,)) for f in FACES if min(dims[f]) >= LABEL - 1e-9}

BINS = {  # asset/scene id -> outer size (m) and sRGB colours
    "bin_blue": dict(size=(14 * IN, 8 * IN, 5 * IN), outside=(40, 57, 73), inside=(223, 178, 112)),       # navy shoe box, kraft inside: lane A
    "bin_red": dict(size=(14 * IN, 9.5 * IN, 5 * IN), outside=(235, 30, 30), inside=(236, 188, 124)),     # red shoe box, kraft inside: lane B
    "reject_bin": dict(size=(13 * IN, 9 * IN, 4.5 * IN), outside=(40, 40, 38), inside=(245, 240, 225)),  # black shoe box, white inside: reject
}
