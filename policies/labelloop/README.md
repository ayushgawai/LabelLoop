# LabelLoop Policy Space deployment (X2Real sim)

Needs `.env` with `WALLX_ENDPOINT`, `WALLX_API_KEY`, agent vars (see `.env.example`).

```
# terminal 1 (policy_space env, repo on PYTHONPATH)
PYTHONPATH=/path/to/LabelLoop policy-space serve /path/to/LabelLoop/policies/labelloop/deploy.yaml --port 8001
# terminal 2 (X2Real / Isaac Sim env)
python manaenv/scripts/x2real_eval.py --board artixon_arm6a_tabletop --server-address 127.0.0.1 --server-port 8001
```
Before trusting a run: replace the placeholder `home_pose` in `configs/ex001_6r.yaml`, then check that sending the current pose back leaves the arm still.
