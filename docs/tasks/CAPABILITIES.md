# Task (Codex): what in the organizers' robot client / WALLX API can make our runs faster, smoother and more reliable?

Read: docs/research/client/*.txt (excerpts of the organizers' robot client: start_desktop.py options, RTC/async inference notes,
file listing), labelloop/wallx_client.py, labelloop/robots/event_server.py, docs/EVENT_RUNBOOK.md, docs/AGENT_CONTEXT.md.
No network, no robot, no keys.

Write docs/research/CAPABILITIES.md (SAVE IT EARLY and update it as you go; partial results must survive if you are cut off):
1. Every client option / API field (infer-type sync|async, infer-params presets, interpolate_multiplier, control_dt,
   action_horizon, RTC, camera resize / JPEG quality, use-history / reset_history, home-on-stop, embodiment variants,
   image wire format, api schema) with what it does, its default, and evidence (file + line).
2. Ranked recommendations for our relay + harness (event_fast mode, LAN WALLX at 0.07 s/call, ~1.4 s per call end to end):
   which settings to change for faster cycles, smoother motion and more reliable grasps, each with the exact command-line
   change to `start_desktop.py ... --model-address 192.168.1.157:8001 ...` or the code change in our relay.
3. Anything that lets the harness tell WALLX more (extra request fields, history, prompts) or stop/abort motion faster.
Keep it under 150 lines. Do not edit any other file.
