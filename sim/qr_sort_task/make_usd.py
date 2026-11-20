"""Write the LabelLoop USDA assets (box per code, 2 decoys, 2 bins, table mat) + textures. Stdlib + PIL + numpy.

  python sim/qr_sort_task/make_usd.py --labels sim/assets --out $MANAENV_ASSET_ROOT/labelloop

Cube 60 mm (CUBE), one OmniPBR material /<root>/Looks/pic, one atlas texture (three stacked squares: top | labeled +Y side | kraft).
NOT LOADED IN ISAAC SIM (no simulator here): check in Kit, esp. UV orientation and rigid-body schema.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

CUBE = 0.06  # keep equal to make_qr_boxes.CUBE_MM / 1000
HX = HY = HZ = CUBE / 2
ROUTES = {"z95112": "420/95112/001", "z90210": "420/90210/001", "z10001": "420/10001/001"}  # keep equal to make_qr_boxes.ROUTES
CODES = [f"{z}{n:03d}" for z in "AB" for n in (1, 2, 3)]


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


def usda(root, body, looks, rigid=True, kinematic=False, mass=0.08):
    api = '    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]\n' if rigid else ""
    props = f"    float physics:mass = {mass}\n" + ("    bool physics:kinematicEnabled = 1\n" if kinematic else "") if rigid else ""
    return f'''#usda 1.0
(
    defaultPrim = "ROOT"
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "ROOT" (
{api})
{{
{props}{body}
    def Scope "Looks"
    {{
{looks}    }}
}}
'''


def box_mesh(mat):
    # (normal-side corners CCW seen from outside) -> faces: +Z top, +Y labeled, others kraft; UV atlas v: kraft 0-1/3, side 1/3-2/3, top 2/3-1
    t = [(-HX, -HY, HZ), (HX, -HY, HZ), (HX, HY, HZ), (-HX, HY, HZ)]
    py = [(HX, HY, -HZ), (-HX, HY, -HZ), (-HX, HY, HZ), (HX, HY, HZ)]
    ny = [(-HX, -HY, -HZ), (HX, -HY, -HZ), (HX, -HY, HZ), (-HX, -HY, HZ)]
    px = [(HX, -HY, -HZ), (HX, HY, -HZ), (HX, HY, HZ), (HX, -HY, HZ)]
    nx = [(-HX, HY, -HZ), (-HX, -HY, -HZ), (-HX, -HY, HZ), (-HX, HY, HZ)]
    nz = [(HX, -HY, -HZ), (-HX, -HY, -HZ), (-HX, HY, -HZ), (HX, HY, -HZ)]
    uv = lambda v0, v1: [(0, v0), (1, v0), (1, v1), (0, v1)]
    faces = [(t, uv(2 / 3, 1)), (py, uv(1 / 3, 2 / 3)), (ny, uv(0, 1 / 3)), (px, uv(0, 1 / 3)), (nx, uv(0, 1 / 3)), (nz, uv(0, 1 / 3))]
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


def cube(name, size, pos, mat, collide=True):
    api = '"PhysicsCollisionAPI", ' if collide else ""
    return f'''    def Cube "{name}" (
        prepend apiSchemas = [{api}"MaterialBindingAPI"]
    )
    {{
        double size = 1
        double3 xformOp:translate = {pos}
        double3 xformOp:scale = {size}
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        rel material:binding = </ROOT/Looks/{mat}>
    }}
'''


def omni(tex=None, color=(0.8, 0.8, 0.8)):
    d = {"float inputs:reflection_roughness_constant": 1.0, "float inputs:metallic_constant": 0.0, "float inputs:specular_level": 0.0}
    if tex:
        d["asset inputs:diffuse_texture"] = f"@{tex}@"
    else:
        d["color3f inputs:diffuse_color_constant"] = color
    return d


def bin_body(color_name, w=0.172, h=0.08):
    t = 0.006
    parts = [cube("floor", (w, w, t), (0, 0, t / 2), color_name)]
    for i, (sx, sy, px_, py_) in enumerate([(t, w, (w - t) / 2, 0), (t, w, -(w - t) / 2, 0), (w, t, 0, (w - t) / 2), (w, t, 0, -(w - t) / 2)]):
        parts.append(cube(f"wall{i}", (sx, sy, h), (px_, py_, h / 2), color_name))
    return "".join(parts)


def mat_texture(path, base, seed):
    noise = np.random.default_rng(seed).normal(0, 10, (1024, 1024, 1))
    Image.fromarray(np.clip(np.array(base) + noise, 0, 255).astype(np.uint8)).save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", default="sim/assets", help="output of sim/make_qr_boxes.py")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out, labels = Path(a.out), Path(a.labels)
    (out / "textures").mkdir(parents=True, exist_ok=True)

    def atlas(top, side, name):
        t, s = Image.open(top), Image.open(side) if side else None
        kraft = Image.new("RGB", t.size, t.getpixel((2, 2)))
        img = Image.new("RGB", (t.width, 3 * t.height))
        img.paste(t, (0, 0))
        img.paste(s.convert("RGB") if s else kraft, (0, t.height))
        img.paste(kraft, (0, 2 * t.height))
        img.save(out / "textures" / f"{name}_atlas.png")

    pic = lambda name: material("pic", omni(f"./textures/{name}_atlas.png"))
    for c in CODES:
        s = f"LL_ZONE-{c[0]}_{c[1:]}"
        atlas(labels / f"{s}_face.png", labels / f"{s}_side.png", f"box_{c.lower()}")
        (out / f"box_{c.lower()}.usda").write_text(usda("box", box_mesh("pic"), pic(f"box_{c.lower()}"), mass=0.08))
    for i in (1, 2):
        atlas(labels / f"decoy_{i}.png", None, f"box_decoy_{i}")
        (out / f"box_decoy_{i}.usda").write_text(usda("box", box_mesh("pic"), pic(f"box_decoy_{i}"), mass=0.08))
    for name, rgb in (("blue", (0.05, 0.25, 0.9)), ("orange", (1.0, 0.45, 0.05))):
        (out / f"bin_{name}.usda").write_text(usda("bin", bin_body("m"), material("m", omni(color=rgb)), kinematic=True, mass=1.0))
    for name, code in ROUTES.items():  # lane-sort parcels: payload 420/<ZIP5>/<seq3>
        s = code.replace("/", "_")
        atlas(labels / f"{s}_face.png", labels / f"{s}_side.png", f"box_{name}")
        (out / f"box_{name}.usda").write_text(usda("box", box_mesh("pic"), pic(f"box_{name}"), mass=0.08))
    atlas(labels / "nolabel.png", None, "box_nolabel")
    (out / "box_nolabel.usda").write_text(usda("box", box_mesh("pic"), pic("box_nolabel"), mass=0.08))
    g = 0.05  # garbage: plain red 50 mm cube, no kraft, no QR
    (out / "box_garbage.usda").write_text(usda("box", cube("g", (g, g, g), (0, 0, g / 2), "m"), material("m", omni(color=(0.85, 0.05, 0.05))), mass=0.06))
    for name, rgb in (("blue", (0.05, 0.25, 0.9)), ("orange", (1.0, 0.45, 0.05))):  # lane mats: 140x140x2 mm, visual only
        (out / f"lane_{name}.usda").write_text(usda("lane", cube("slab", (0.14, 0.14, 0.002), (0, 0, 0.001), "m", collide=False),
                                                    material("m", omni(color=rgb)), kinematic=True, mass=1.0))
    (out / "reject_basket.usda").write_text(usda("basket", bin_body("m", 0.25, 0.15), material("m", omni(color=(0.22, 0.22, 0.24))),
                                                 kinematic=True, mass=2.0))
    (out / "reject_bin.usda").write_text(usda("bin", bin_body("m", 0.16, 0.10), material("m", omni(color=(0.1, 0.1, 0.11))), kinematic=True, mass=1.0))
    mat_texture(out / "textures" / "mat_1.png", (60, 60, 64), 1)
    mat_texture(out / "textures" / "mat_2.png", (200, 205, 190), 2)
    (out / "mat.usda").write_text(usda("mat", cube("slab", (0.5, 0.9, 0.001), (0, 0, 0.0005), "top", collide=False),
                                       material("top", omni("./textures/mat_1.png")), rigid=False))
    print(f"wrote {len(list(out.glob('*.usda')))} usda + textures to {out}")


if __name__ == "__main__":
    main()
