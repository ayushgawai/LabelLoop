"""LabelLoop QR-sort assets, success term and disturbance event for X2Real.

Install: copy to <X2REAL>/manaenv/assets/objects/labelloop_qr_sort.py (scene YAML preloads it via
background.asset_module). Follows manaenv/assets/objects/move_bottle_to_bin.py.
USD files + textures come from make_usd.py into $MANAENV_ASSET_ROOT/labelloop/.
"""
import os

import torch
from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.assets.object_library import LibraryObject
from isaaclab_arena.assets.register import register_asset

from manaenv.assets.resolver import resolve_runtime_asset_reference

ROOT = "assets/labelloop"
_ALL = ("pz", "nz", "py", "ny", "px", "nx")
# parcel id -> faces that can hold its label; one asset box_<id>_<face> per face (props.variants; tests/test_layout.py keeps it equal)
LABELLED = {"z95112": _ALL, "z90210": _ALL, "z10001": ("pz", "nz"), "z94110": _ALL, "z90015": _ALL, "z60601": _ALL, "z95014": _ALL,
            "bhindi": ("pz", "nz"), "lobar": ("pz", "nz"), "miro": ("pz", "nz")}
PARCELS = ("nolabel", "garbage", "wallet", "plainbox", *(f"{b}_{f}" for b, fs in LABELLED.items() for f in fs))


def _usd(name):
    return resolve_runtime_asset_reference(f"{ROOT}/{name}.usda")


class _Rigid(LibraryObject):
    object_type = ObjectType.RIGID

    def _generate_rigid_cfg(self):
        from isaaclab.assets import RigidObjectCfg
        from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

        return self._add_initial_pose_to_cfg(RigidObjectCfg(
            prim_path=self.prim_path, spawn=UsdFileCfg(usd_path=self.usd_path, scale=self.scale, activate_contact_sensors=True)))


def _register(name, usd, tags):
    register_asset(type(name, (_Rigid,), {"name": name, "usd_path": _usd(usd), "tags": ["object", *tags]}))


for _n in PARCELS:
    _register(f"labelloop_box_{_n}", f"box_{_n}", ["pickable", "qr_box"])
for _b in ("bin_blue", "bin_red", "reject_bin"):  # the three shoe boxes (props.py)
    _register(f"labelloop_{_b}", _b, ["container", "bin"])
_register("labelloop_table_mat", "labelloop_table_mat", [])  # not "pickable"/"container": no semantic role


def register_assets() -> None:
    """Preload hook used by the scene factory (registration happens at import)."""


def _pos(env, name):
    d = env.scene[name].data
    return d.root_pos_w.torch, d.root_lin_vel_w.torch


def _consecutive(env, ok, n, attr):
    count = getattr(env, attr, None)
    if count is None or count.shape[0] != env.num_envs:
        count = torch.zeros(env.num_envs, dtype=torch.int64, device=env.device)
    count = (count + 1) * ok
    setattr(env, attr, count)
    return count >= n


def in_bin(env, obj, bin_, hx, hy, z_max, vel_max=0.5):
    """Per-step bool: obj centre inside the bin cavity (|dx| <= hx, |dy| <= hy, 0 <= dz <= z_max relative to the bin origin = centre
    of its underside) and at rest. The bins are placed with yaw 0 (or 180) so their length lies along env x."""
    p, v = _pos(env, obj)
    rel = p - _pos(env, bin_)[0]
    return ((rel[:, 0].abs() <= hx) & (rel[:, 1].abs() <= hy) & (rel[:, 2] >= 0.0) & (rel[:, 2] <= z_max)
            & (v.norm(dim=-1) <= vel_max))


def lanes_sorted(env, *, assignments, bins, vel_max=0.5, consecutive_steps=5):
    """Termination: assignments [[objects, bin id], ...]; bins {bin id: [hx, hy, z_max]} (cavity, from build_task.success_region).
    A group is ok when any of its objects that is NOT parked (env z >= -2 m) rests in its bin; a group whose members are all parked is
    not ok (a labelled parcel has one variant per label face, only the one the human dropped is unparked). Every group for consecutive_steps."""
    ok = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    for objs, target in assignments:
        ok &= torch.stack([in_bin(env, o, target, *bins[target], vel_max) & ~_parked(env, o) for o in objs]).any(dim=0)
    return _consecutive(env, ok, consecutive_steps, "_labelloop_lanes_count")


def never(env):
    """Termination that never fires (calibration case 31 has nothing to sort; X2Real requires a success term per stage)."""
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def _parked(env, name):
    return _pos(env, name)[0][:, 2] - env.scene.env_origins[:, 2] < -2.0


_GEN = {}


def _gen(device):
    """Own generator seeded from os.urandom once per process: the env's global seed is fixed, which made every episode's arrival identical."""
    if device not in _GEN:
        _GEN[device] = torch.Generator(device=device).manual_seed(int.from_bytes(os.urandom(8), "little") >> 1)
    return _GEN[device]


def disturb(env, env_ids, *, object_name, to_xy_range, yaw_range=None, z=None, avoid=(), slide_v=0.0):
    """Interval event (``mode: interval, interval_range_s: [t, t], is_global_time: true``): a human slides a parcel in.

    object_name: one name, or a list (the label-face variants of one parcel): per env ONE uniformly random member is dropped, and only if
    none of the list has been dropped yet (all still parked, env z < -2 m), so the repeated interval firing cannot drop a second one.
    Teleports it to a random xy/yaw in to_xy_range [[x0,x1],[y0,y1]] (env frame; + optional yaw_range [lo,hi] rad, z) each
    episode: of 32 uniform candidates the one farthest from the ``avoid`` objects, so it never lands inside another parcel.
    slide_v > 0: a swipe from the left: placed 0.08 m further left (env +y) and given slide_v m/s toward the robot's right (env -y).
    Prints one line per drop: ``[labelloop] drop <object> env <i>``.
    """
    names = [object_name] if isinstance(object_name, str) else list(object_name)
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids.to(env.device).reshape(-1)
    rel_z = torch.stack([_pos(env, o)[0][ids, 2] - env.scene.env_origins[ids, 2] for o in names], dim=1)  # (E, K)
    ids = ids[(rel_z < -2.0).all(dim=1)]
    if len(ids) == 0:
        return
    pick = torch.randint(len(names), (len(ids),), device=env.device, generator=_gen(env.device))
    for k, name in enumerate(names):
        _drop(env, name, ids[pick == k], to_xy_range, yaw_range, z, avoid, slide_v)


def _drop(env, object_name, ids, to_xy_range, yaw_range, z, avoid, slide_v):
    if len(ids) == 0:
        return
    asset = env.scene[object_name]
    d = asset.data
    pose = torch.cat([d.root_pos_w.torch, d.root_quat_w.torch], dim=-1)[ids].clone()
    n, u = len(ids), lambda lo, hi: lo + (hi - lo) * torch.rand(n, device=env.device, generator=_gen(env.device))
    cand = torch.stack([lo + (hi - lo) * torch.rand(n, 32, device=env.device, generator=_gen(env.device)) for lo, hi in to_xy_range], dim=-1)  # (n, 32, 2)
    if avoid:
        others = torch.stack([env.scene[o].data.root_pos_w.torch[ids, :2] - env.scene.env_origins[ids, :2] for o in avoid], dim=1)  # (n, M, 2)
        best = torch.cdist(cand, others).min(dim=-1).values.argmax(dim=-1)
    else:
        best = torch.zeros(n, dtype=torch.int64, device=env.device)
    xy = cand[torch.arange(n, device=env.device), best]
    if slide_v:
        xy[:, 1] += 0.08
    pose[:, :3] = env.scene.env_origins[ids] + torch.cat([xy, torch.full((n, 1), float(z), device=env.device)], dim=-1)
    yaw = u(*yaw_range) if yaw_range else torch.zeros(n, device=env.device)
    pose[:, 3:] = torch.stack([torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2), torch.cos(yaw / 2)], dim=-1)  # XYZW
    asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
    if hasattr(asset, "write_root_velocity_to_sim_index"):
        vel = torch.zeros(len(ids), 6, device=env.device)
        vel[:, 1] = -slide_v
        asset.write_root_velocity_to_sim_index(root_velocity=vel, env_ids=ids)
    for i in ids.tolist():
        print(f"[labelloop] drop {object_name} env {i}", flush=True)
