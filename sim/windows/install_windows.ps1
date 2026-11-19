# Native-Windows install of X2Real (Isaac Sim 6.0.1 stack). Idempotent-ish: re-run is safe.
# Usage (detached): powershell -NoProfile -ExecutionPolicy Bypass -File C:\x2\scripts\install_windows.ps1
$ErrorActionPreference = 'Continue'
$env:OMNI_KIT_ACCEPT_EULA = 'YES'; $env:PYTHONIOENCODING = 'utf-8'
$R = 'C:\x2\x2real'; $IL = "$R\3rd\isaaclabarena\submodules\IsaacLab"; $A = "$R\3rd\isaaclabarena"
$V = 'C:\x2\venv'; $py = "$V\Scripts\python.exe"
New-Item -ItemType Directory -Force C:\x2\logs | Out-Null
function Run($desc, [scriptblock]$b) {
  "=== $desc $(Get-Date -f s)"; & $b
  if ($LASTEXITCODE) { "FAILED: $desc ($LASTEXITCODE)"; "EXIT $LASTEXITCODE"; exit $LASTEXITCODE }
}
# 1. sources (Isaac-GR00T submodule is an internal GitLab repo: not needed, skipped)
if (-not (Test-Path "$R\.git")) { Run 'clone x2real' { git clone https://github.com/X-Square-Robot/x2real $R } }
Set-Location $R
Run 'submodules' { git submodule update --init 3rd/policy_space 3rd/isaaclabarena }
Set-Location $A
Run 'IsaacLab submodule' { git submodule update --init submodules/IsaacLab }
Set-Location "$IL"; if (-not (Test-Path "$IL\isaaclab.bat")) { git checkout -f HEAD 2>&1 | Out-Null }   # an aborted recursive update leaves an empty worktree
# 2. venv
if (-not (Test-Path $py)) { Run 'venv' { uv venv $V --python 3.12 } }
$env:VIRTUAL_ENV = $V; $env:PATH = "$V\Scripts;$env:PATH"
Run 'pip/uv' { uv pip install --python $py --upgrade pip }
# 2b. Windows patches (see README.md). Skipped per-repo if already applied.
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
foreach ($p in @(@('x2real.patch',$R), @('arena.patch',$A), @('isaaclab.patch',$IL))) {
  git -C $p[1] apply --ignore-whitespace --reverse --check "$here\$($p[0])" 2>$null
  if ($LASTEXITCODE) { git -C $p[1] apply --ignore-whitespace "$here\$($p[0])"; "applied $($p[0]) rc=$LASTEXITCODE" } else { "already applied: $($p[0])" }
}
# 3. Isaac Sim 6.0.1.0 + torch + Isaac Lab + Newton (Isaac Lab's own CLI is cross-platform)
Set-Location $IL
Run 'isaaclab -i' { cmd /c "isaaclab.bat -i all,isaacsim" }
# 4. Arena deps (setup_arena_local.sh steps 2-7)
Set-Location $A
Run 'arena deps' { uv pip install --python $py pytest typing_extensions onnxruntime vuer "huggingface-hub[cli]" qpsolvers==4.8.1 }
Run 'arena' { uv pip install --python $py -e . }
# 5. policy-space protocol + x2real (install.py editable step), Windows runtime-wheel profile
Set-Location C:\x2   # NOT the arena dir: its pyproject [tool.uv] pins Newton on the internal GitLab
Run 'newton pins' { uv pip install --python $py --no-deps "newton[importers] @ git+https://github.com/X-Square-Robot/Newton.git@x2real" "newton-usd-schemas @ git+https://github.com/X-Square-Robot/Newton.git@x2real#subdirectory=extern/newton-usd-schemas" }
Run 'policy-space protocol' { uv pip install --python $py --no-deps -e "$R\3rd\policy_space\packages\protocol" }
Run 'policy-space server (CLI)' { uv pip install --python $py --no-deps -e "$R\3rd\policy_space" }
Run 'coverage pin (numba import contract, as x2real_runtime_wheels.py does)' { uv pip install --python $py --no-deps coverage==7.6.1 }
# NOTE: x2real_runtime_wheels.py is skipped on Windows (only rewrites pip metadata; hashes are for the Linux wheels).
Set-Location $R
Run 'x2real' { uv pip install --python $py -e . }
"EXIT 0"
