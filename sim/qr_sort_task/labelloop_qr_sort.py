"""LabelLoop QR-sort assets, success term and disturbance event for X2Real.

Install: copy to <X2REAL>/manaenv/assets/objects/labelloop_qr_sort.py (scene YAML preloads it via
background.asset_module). Follows manaenv/assets/objects/move_bottle_to_bin.py. NOT RUN (no Isaac Sim here).
USD files + textures come from make_usd.py into $MANAENV_ASSET_ROOT/labelloop/.
"""
import torch
from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.assets.object_library import LibraryObject
from isaaclab_arena.assets.register import register_asset

from manaenv.assets.resolver import resolve_runtime_asset_reference

ROOT = "assets/labelloop"
CODES = [f"{z}{n:03d}" for z in "AB" for n in (1, 2, 3)]  # a001..b003, matches sim/make_qr_boxes.py
ZONE_BIN = {"A": "bin_blue", "B": "bin_orange"}


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


for _c in CODES:
    _register(f"labelloop_box_{_c.lower()}", f"box_{_c.lower()}", ["pickable", "qr_box"])
for _i in (1, 2):
    _register(f"labelloop_decoy_{_i}", f"box_decoy_{_i}", ["pickable", "decoy"])
_register("labelloop_bin_blue", "bin_blue", ["container", "bin"])
_register("labelloop_bin_orange", "bin_orange", ["container", "bin"])
_register("labelloop_mat", "mat", ["furniture"])


def register_assets() -> None:
    """Preload hook used by the scene factory (registration happens at import)."""


def _pos(env, name):
    d = env.scene[name].data
    return d.root_pos_w.torch, d.root_lin_vel_w.torch


def boxes_sorted(env, *, assignments, bins, xy=0.07, z_min=-0.02, z_max=0.15, vel_max=0.5, consecutive_steps=5):
    """Termination (use as ``_func: manaenv.assets.objects.labelloop_qr_sort.boxes_sorted``).

    assignments: {scene_object_id: bin_object_id | None}. A box with a bin must rest in it; a box mapped to
    None (unlabeled decoy) must be in none of ``bins``. Mirrors terminations.classified_object_on_color_target.
    """
    ok = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    bin_pos = {b: _pos(env, b)[0] for b in bins}
    for box, target in assignments.items():
        p, v = _pos(env, box)
        inside = {}
        for b, bp in bin_pos.items():
            rel = p - bp
            inside[b] = (rel[:, :2].norm(dim=-1) <= xy) & (rel[:, 2] >= z_min) & (rel[:, 2] <= z_max) & (v.norm(dim=-1) <= vel_max)
        if target is None:
            for b in bins:
                ok &= ~inside[b]
        else:
            ok &= inside[target]
    attr = "_labelloop_sorted_count"
    count = getattr(env, attr, None)
    if count is None or count.shape[0] != env.num_envs:
        count = torch.zeros(env.num_envs, dtype=torch.int64, device=env.device)
    count = (count + 1) * ok
    setattr(env, attr, count)
    return count >= consecutive_steps


def disturb(env, env_ids, *, object_name, delta_xyz=None, to_xyz=None):
    """Interval event (``mode: interval, interval_range_s: [t, t], is_global_time: true``): human moves/adds a box.

    delta_xyz: shift the box by this vector (env frame). to_xyz: teleport it (use for a box parked far away at
    reset, i.e. "added mid-episode"). ponytail: an interval term fires repeatedly; one-shot-ness relies on
    to_xyz being idempotent, and delta_xyz repeating every tick (use a long interval_range_s).
    """
    asset = env.scene[object_name]
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids.to(env.device).reshape(-1)
    d = asset.data
    pose = torch.cat([d.root_pos_w.torch, d.root_quat_w.torch], dim=-1)[ids].clone()
    if to_xyz is not None:
        pose[:, :3] = env.scene.env_origins[ids] + torch.tensor(to_xyz, device=env.device)
    if delta_xyz is not None:
        pose[:, :3] += torch.tensor(delta_xyz, device=env.device)
    asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
    if hasattr(asset, "write_root_velocity_to_sim_index"):
        asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros(len(ids), 6, device=env.device), env_ids=ids)
