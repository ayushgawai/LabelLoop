# Event task files (organizer WebUI)

- `local_labelloop_pipeline_test.json` - 1 case, "put the box into the blue bin" (pipeline smoke test).
- `local_labelloop_sort.json` - LabelLoop demo self-test, 6 cases of increasing difficulty, 10 points each.

Both point at our relay (`ws://192.168.1.157:8001`, config `labelloop_relay`).

## Use on the robot

1. Start the LabelLoop relay on 192.168.1.157:8001 and check it is listening.
2. Copy the task file(s):
   `scp results/event/tasks/local_labelloop_*.json xr@192.168.1.21:~/diversity/`
3. Open http://192.168.1.21/robot/home and select the task (e.g. `local_labelloop_sort`).
4. Set up the scene per case `env_description`, then run (timeout 480 s per case).
