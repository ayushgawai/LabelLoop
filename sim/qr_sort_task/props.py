"""The real props, measured by hand: one place for make_usd.py (geometry, colours) and build_task.py (layout, success check).

Bins are the three open shoe boxes (lids off) the robot sorts into. Size = outer length x width x height, measured in inches.
Length runs away from the robot (env x), width runs across the table (env y). Colours are sRGB medians sampled from the photos
(room light, phone camera); calibrate the harness HSV ranges on sim/real head-camera frames, not on these numbers.
"""
IN = 0.0254
WALL = 0.005  # modelled wall/floor thickness (m): real board is ~2-3 mm, 5 mm keeps a falling parcel from tunnelling through

GRIPPER_OPEN = 0.0975  # G gripper fully open between the fingers (2 x 48.75 mm, robot URDF): a parcel's grasp width must be under it
CUBE = (0.06, 0.06, 0.06)  # sim stand-in for a parcel that has no measured prop yet
FACES = ("pz", "nz", "py", "ny", "px", "nx")  # box faces +z top, -z bottom, +-y sides, +-x ends (nx faces the robot)

# Measured parcels, keyed by the parcel id build_task.py uses (z95112 = lane A, QR 420/95112/001; z90210 = lane B; z10001 = unknown
# ZIP -> reject). size = the parcel as it ARRIVES in the induction zone: x away from the robot, y across the table, z up (fingers close
# sideways, so y is the grasp width). yaw = how far (rad) the human may leave it turned; mass = spawn range (kg, guessed: weigh them).
# color = body sRGB from the photos. tape = a stripe of tape / print looping round the parcel: centred `at` mm from the middle along
# `axis`, `mm` wide along it, on the four faces parallel to that axis. labels = faces that carry the 50 mm QR label (centred).
# patches = printed rectangles: face, centre `at` (u, v) mm from the face centre in its tile axes, size `mm` (u, v), drawn in order.
# finish = OmniPBR metallic / roughness for a shiny parcel (default matte paper). Not listed = still the 60 mm sim cube (red 50 mm cube
# for garbage).
PARCELS = {
    # package 1: Amazon box 9 x 6 x 3.5 in, stood on its long edge, long side pointing away from the robot: the only pose the gripper
    # can close on (3.5 in = 89 mm across). Seam tape = belt round it at mid height. Label on top and on the end facing the robot.
    "z95112": dict(size=(9 * IN, 3.5 * IN, 6 * IN), mass=(0.10, 0.20), yaw=0.1, color=(204, 166, 118),
                   tape=dict(axis="z", mm=72, at=0, rgb=(33, 30, 28)), labels=("pz", "nx")),
    # package 2: Amazon box 11 x 8.1 x 4 in, lying flat, long side ACROSS the table (lengthwise it does not fit in front of the bins).
    # NOT graspable by the G gripper: thinnest side 4 in = 102 mm. Seam tape along its length over the top and down both ends.
    "z90210": dict(size=(8.1 * IN, 11 * IN, 4 * IN), mass=(0.15, 0.30), yaw=0.1, color=(188, 157, 123),
                   tape=dict(axis="x", mm=72, at=0, rgb=(40, 40, 40)), labels=("pz", "nx")),
    # package 3: Amazon padded mailer 10 x 8.5 x 0.5 in, lying flat, long side across. NOT graspable (13 mm thick, 216-254 mm wide),
    # and rigid in sim (the real one bends). Off-white; blue tear band across it 72 mm from the middle. Label on top only.
    "z10001": dict(size=(8.5 * IN, 10 * IN, 0.5 * IN), mass=(0.05, 0.10), yaw=0.1, color=(231, 222, 205),
                   tape=dict(axis="y", mm=60, at=72, rgb=(49, 100, 179)), labels=("pz",)),
    # package 4 = the garbage item (a retail box with no shipping label -> black bin): perfume box 6 x 4 x 2 in, brushed light-blue foil,
    # stood on its long edge like package 1 (2 in = 51 mm across, ~23 mm spare per finger). Grey-bordered white name label on one big
    # side. The real box also has a small authenticity sticker with its own QR; not modelled. Mass guessed for a boxed 100 ml bottle.
    "garbage": dict(size=(6 * IN, 2 * IN, 4 * IN), mass=(0.25, 0.40), yaw=0.3, color=(109, 128, 145), labels=(),
                    finish=dict(metallic=0.5, roughness=0.35),
                    patches=[dict(face="py", at=(-30, 12), mm=(58, 58), rgb=(175, 178, 182)),
                             dict(face="py", at=(-30, 12), mm=(40, 28), rgb=(246, 240, 224))]),
}

BINS = {  # asset/scene id -> outer size (m) and sRGB colours
    "bin_blue": dict(size=(14 * IN, 8 * IN, 5 * IN), outside=(40, 57, 73), inside=(223, 178, 112)),       # navy shoe box, kraft inside: lane A
    "bin_red": dict(size=(14 * IN, 9.5 * IN, 5 * IN), outside=(235, 30, 30), inside=(236, 188, 124)),     # red shoe box, kraft inside: lane B
    "reject_bin": dict(size=(13 * IN, 9 * IN, 4.5 * IN), outside=(40, 40, 38), inside=(245, 240, 225)),  # black shoe box, white inside: reject
}
