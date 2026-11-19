# X2Real (Isaac Sim 6.0.1) on native Windows

Verified on Windows 11, RTX 5080 Laptop, driver 592.02. Layout: `C:\x2\x2real` (repo + submodules),
`C:\x2\venv` (uv, Python 3.12), assets `C:\x2\x2real-assets` (`MANAENV_ASSET_ROOT`).

## Install
`install_windows.ps1` (re-runnable): clones x2real + submodules, applies the patches below, then
`isaaclab.bat -i all,isaacsim` (Isaac Sim 6.0.1.0, torch 2.11+cu128, Isaac Lab, Newton@x2real),
arena deps, policy-space, `pip install -e` of everything. About 15 min on this machine (Isaac Sim wheels ~10 GB).
Run detached: `scripts\launch.ps1 C:\x2\scripts\install_windows.ps1 C:\x2\logs\install.log`.

## Patches (`*.patch`, applied with `git apply --ignore-whitespace`)
- `isaaclab.patch`: IsaacLab `pyproject.toml` pins Newton to an internal GitLab (`ssh://gitlab.zbl.local`);
  point it at `git+https://github.com/X-Square-Robot/Newton.git@x2real` (same repo/branch `install.py` uses).
- `arena.patch`: `isaaclab_arena/embodiments/__init__.py` wraps each embodiment import in `try/except ImportError`.
  The G1/GR1 humanoid code imports `pinocchio`, which has no Windows wheel; X2 robots do not need it.
- `x2real.patch`:
  - `manaenv/controllers/batched_fk.py`: only user of pinocchio for X2 robots (URDF chain extraction at init).
    Falls back to a ~50 line numpy URDF parser when `pinocchio` is not importable (same offsets/axes, +X/+Y/+Z revolute axes only).
  - `manaenv/pipeline/evaluation/report.py`: skip directory `fsync` on Windows (`os.open(dir)` raises PermissionError).
- Not applied on Windows: `x2real_runtime_wheels.py` (only rewrites pip metadata, hashes are Linux wheels).
  `pip check` therefore reports isaacsim-core/kernel pin conflicts (newton, mujoco, websockets...); runtime is fine.
  `coverage==7.6.1` is pinned by hand as that script does.
- Isaac-GR00T submodule (internal GitLab) is skipped; not needed.

## Run (all detached; SSH sessions kill children otherwise)
```
C:\x2\scripts\launch.ps1 C:\x2\scripts\run_policy_server.ps1 C:\x2\logs\ps_server.log   # port 8001
C:\x2\scripts\launch.ps1 C:\x2\scripts\run_eval.ps1          C:\x2\logs\eval.log        # edit args inside
```
Scripts live in this directory (`launch.ps1`, `run_policy_server.ps1`, `run_eval.ps1`); copy to `C:\x2\scripts\`.
Stop server: `Stop-Process -Id (Get-NetTCPConnection -LocalPort 8001 -State Listen).OwningProcess`.
Headless is the default (no `--headless` flag). Outputs: `C:\x2\x2real\outputs\x2real_eval\<board>\benchmark\`.
Rendering works when launched via CIM (session 0); no need for the interactive-session fallback.

## Timings
First eval launch: ~70 s to scene build (shader caches warm quickly), 1500-frame episode in ~160 s wall, ~290 s total.

## Known issues
- No pinocchio/pink on Windows: G1/GR1 embodiments and Isaac Lab pink IK unavailable; other robots untested.
- Asset restore verifier (`restore_assets.py`) fails its final file-set check on Windows: the bundle has
  `wooden_table_02_Normal.png` and `.PNG` in one dir (identical sha256), which collide on a case-insensitive FS.
  Content is intact; all other files hash-verified.
- Zero-action policy scores 0 (time_out at `place_fruit_1`); that is the expected smoke-test result.
- Console needs `PYTHONIOENCODING=utf-8` (set in the scripts).
