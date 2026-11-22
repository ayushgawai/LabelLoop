@echo off
rem Scenario 4 with WALLX, saved demo recipe scenarios/w4x/saved/sc4_wallx_v1 (replayed pick + handover, WALLX places with the left arm).
rem Run from anywhere:  scripts\run_sc4_wallx_demo.cmd        (add --check to only print the relay address)
rem Before: Bhindi box standing on its long narrow edge just in front of zone 2's front-left corner
rem         (runs/manual/sc4_demo_start_pose.jpg); arms at rest; no other session on the robot.
cd /d "%~dp0.."
set IP=
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4" ^| findstr /c:"192.168.1."') do set IP=%%a
set IP=%IP: =%
if "%IP%"=="" (echo laptop is not on the robot LAN 192.168.1.x & exit /b 1)
set LL_RELAY_ADDR=%IP%:8001
echo relay address for the robot: %LL_RELAY_ADDR%
if "%1"=="--check" exit /b 0
if not exist runs mkdir runs
if not exist runs\empty_stdin.txt type nul > runs\empty_stdin.txt
.venv\Scripts\python scripts\run_trial.py --scenario scenarios/w4x/saved/sc4_wallx_v1/recipe.json < runs\empty_stdin.txt
