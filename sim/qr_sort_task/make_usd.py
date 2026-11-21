"""Write the LabelLoop USDA assets (lane-sort parcels, the three shoe-box bins from props.py) + textures. Stdlib + PIL.

  python sim/qr_sort_task/make_usd.py --labels sim/assets --out $MANAENV_ASSET_ROOT/labelloop

Parcel = one textured box mesh (origin at its centre), one OmniPBR material /<root>/Looks/pic, one atlas texture. `nolabel` is a plain
60 mm kraft cube (CUBE); every measured parcel (props.PARCELS) has its real size and an atlas of its six face tiles from make_qr_boxes.py,
one asset box_<variant>.usda per props.variants (a labelled parcel: one per face that can hold its label).
Bin = open box, origin at the centre of its underside, length along x; every wall and the floor is two layers (outside colour, inside colour).
"""
import argparse
from pathlib import Path

from PIL import Image

from props import BINS, FACES, PARCELS, WALL, variants

CUBE = 0.06  # keep equal to make_qr_boxes.CUBE_MM / 1000
HX = HY = HZ = CUBE / 2
CUBE_UV = {"pz": (0, 2 / 3, 1, 1), "py": (0, 1 / 3, 1, 2 / 3), "ny": (0, 0, 1, 1 / 3), "px": (0, 0, 1, 1 / 3), "nx": (0, 0, 1, 1 / 3),
           "nz": (0, 0, 1, 1 / 3)}  # face -> (u0, v0, u1, v1) in the cube atlas


def material(path, shader_inputs):
    ins = "\n".join(f"            {k} = {v}" for k, v in shader_inputs.items())
    return f'''        def Material "{path}"
        {{
            token outputs:mdl:surface.connect = </ROOT/Looks/{path}/Shader.outputs:out>
            def Shader "Shader"
            {{
                uniform token info:implementationSource = "sourceAsset"
                uniform asset info:mdl:sourceAsset = @OmniPBR.mdl@
                uniform token info:mdl:sourceAsset:subIdentifier = "OmniPBR"
{ins}
                token outputs:out
            }}
        }}
'''


def usda(root, body, looks, kinematic=False, mass=0.08):
    kin = "    bool physics:kinematicEnabled = 1\n" if kinematic else ""
    return f'''#usda 1.0
(
    defaultPrim = "ROOT"
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "ROOT" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
)
{{
    float physics:mass = {mass}
{kin}{body}
    def Scope "Looks"
    {{
{looks}    }}
}}
'''


def box_mesh(mat, half=(HX, HY, HZ), uv=CUBE_UV):
    # (normal-side corners CCW seen from outside, corner 0 = tile bottom-left) per face; uv = face -> (u0, v0, u1, v1) atlas rectangle.
    # Tile axes: pz u +x / v +y; nz u -x / v +y; py u -x, ny u +x, px u +y, nx u -y, all with v up.
    HX, HY, HZ = half
    t = [(-HX, -HY, HZ), (HX, -HY, HZ), (HX, HY, HZ), (-HX, HY, HZ)]
    py = [(HX, HY, -HZ), (-HX, HY, -HZ), (-HX, HY, HZ), (HX, HY, HZ)]
    ny = [(-HX, -HY, -HZ), (HX, -HY, -HZ), (HX, -HY, HZ), (-HX, -HY, HZ)]
    px = [(HX, -HY, -HZ), (HX, HY, -HZ), (HX, HY, HZ), (HX, -HY, HZ)]
    nx = [(-HX, HY, -HZ), (-HX, -HY, -HZ), (-HX, -HY, HZ), (-HX, HY, HZ)]
    nz = [(HX, -HY, -HZ), (-HX, -HY, -HZ), (-HX, HY, -HZ), (HX, HY, -HZ)]
    quad = lambda u0, v0, u1, v1: [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
    faces = [(t, quad(*uv["pz"])), (py, quad(*uv["py"])), (ny, quad(*uv["ny"])), (px, quad(*uv["px"])), (nx, quad(*uv["nx"])),
             (nz, quad(*uv["nz"]))]
    pts = ", ".join(str(p) for f, _ in faces for p in f)
    uvs = ", ".join(str(u) for _, f in faces for u in f)
    return f'''    def Mesh "Geom" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "MaterialBindingAPI"]
    )
    {{
        int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
        int[] faceVertexIndices = [{", ".join(map(str, range(24)))}]
        point3f[] points = [{pts}]
        texCoord2f[] primvars:st = [{uvs}] (interpolation = "faceVarying")
        uniform token physics:approximation = "convexHull"
        uniform bool doubleSided = 1
        rel material:binding = </ROOT/Looks/pic>
    }}
'''


def cube(name, size, pos, mat):
    return f'''    def Cube "{name}" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "MaterialBindingAPI"]
    )
    {{
        double size = 1
        double3 xformOp:translate = {pos}
        double3 xformOp:scale = {size}
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        rel material:binding = </ROOT/Looks/{mat}>
    }}
'''


def omni(tex=None, color=(0.8, 0.8, 0.8), metallic=0.0, roughness=1.0):
    """OmniPBR inputs; default matte paper (no specular). metallic > 0 (foil) also turns specular on."""
    d = {"float inputs:reflection_roughness_constant": roughness, "float inputs:metallic_constant": metallic,
         "float inputs:specular_level": 0.5 if metallic else 0.0}
    if tex:
        d["asset inputs:diffuse_texture"] = f"@{tex}@"
    else:
        d["color3f inputs:diffuse_color_constant"] = color
    return d


def bin_body(size, t=WALL):
    """Open box L x W x H (x, y, z), walls/floor t thick: an outer shell in material "out" and an inner liner in "in", no overlaps."""
    L, W, H = size
    s = t / 2  # each layer is half the wall
    r = lambda v: round(v, 5)  # noqa: E731
    shell = [("floor", (L - 2 * s, W - 2 * s, s), (0, 0, s / 2)),
             ("end0", (s, W, H), (L / 2 - s / 2, 0, H / 2)), ("end1", (s, W, H), (-(L / 2 - s / 2), 0, H / 2)),
             ("side0", (L - 2 * s, s, H), (0, W / 2 - s / 2, H / 2)), ("side1", (L - 2 * s, s, H), (0, -(W / 2 - s / 2), H / 2))]
    liner = [("floor", (L - 2 * t, W - 2 * t, s), (0, 0, s + s / 2)),
             ("end0", (s, W - 2 * s, H - s), (L / 2 - 1.5 * s, 0, s + (H - s) / 2)),
             ("end1", (s, W - 2 * s, H - s), (-(L / 2 - 1.5 * s), 0, s + (H - s) / 2)),
             ("side0", (L - 2 * t, s, H - s), (0, W / 2 - 1.5 * s, s + (H - s) / 2)),
             ("side1", (L - 2 * t, s, H - s), (0, -(W / 2 - 1.5 * s), s + (H - s) / 2))]
    return "".join(cube(f"{layer}_{n}", tuple(map(r, sz)), tuple(map(r, p)), layer)
                   for layer, parts in (("out", shell), ("in", liner)) for n, sz, p in parts)


def srgb_to_linear(rgb):
    """0-255 sRGB -> linear 0-1, what an OmniPBR colour constant expects (a texture would be decoded from sRGB instead)."""
    c = [v / 255 for v in rgb]
    return tuple(round(v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4, 4) for v in c)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", default="sim/assets", help="output of sim/make_qr_boxes.py")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out, labels = Path(a.out), Path(a.labels)
    (out / "textures").mkdir(parents=True, exist_ok=True)

    def kraft_atlas(top, name):  # cube atlas: three stacked squares, all plain
        t = Image.open(top)
        img = Image.new("RGB", (t.width, 3 * t.height), t.getpixel((2, 2)))
        img.paste(t, (0, 0))
        img.save(out / "textures" / f"{name}_atlas.png")

    def tile_atlas(name):
        """Stack a measured parcel's six face tiles (make_qr_boxes.py) into one atlas; -> face -> (u0, v0, u1, v1)."""
        tiles = {f: Image.open(labels / f"box_{name}_{f}.png").convert("RGB") for f in FACES}
        W, H = max(t.width for t in tiles.values()), sum(t.height for t in tiles.values())
        img, uv, y = Image.new("RGB", (W, H), tiles["nz"].getpixel((0, 0))), {}, 0
        for f, t in tiles.items():  # image row y = atlas v 1 - y / H
            img.paste(t, (0, y))
            uv[f] = (0, round(1 - (y + t.height) / H, 6), round(t.width / W, 6), round(1 - y / H, 6))
            y += t.height
        img.save(out / "textures" / f"box_{name}_atlas.png")
        return uv

    pic = lambda name, **finish: material("pic", omni(f"./textures/{name}_atlas.png", **finish))  # noqa: E731
    for name, b in BINS.items():  # shoe-box bins (props.py)
        looks = material("out", omni(color=srgb_to_linear(b["outside"]))) + material("in", omni(color=srgb_to_linear(b["inside"])))
        (out / f"{name}.usda").write_text(usda("bin", bin_body(b["size"]), looks, kinematic=True, mass=0.3))
    for base in PARCELS:  # measured props, one asset per label-face variant
        for vid, p in variants(base).items():
            mesh = box_mesh("pic", tuple(v / 2 for v in p["size"]), tile_atlas(vid))
            (out / f"box_{vid}.usda").write_text(usda("box", mesh, pic(f"box_{vid}", **p.get("finish", {})), mass=round(sum(p["mass"]) / 2, 3)))
    kraft_atlas(labels / "nolabel.png", "box_nolabel")
    (out / "box_nolabel.usda").write_text(usda("box", box_mesh("pic"), pic("box_nolabel"), mass=0.08))
    print(f"wrote {len(list(out.glob('*.usda')))} usda + textures to {out}")


if __name__ == "__main__":
    main()
