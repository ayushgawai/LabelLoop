import os

import yaml


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_env(path=".env"):
    """KEY=VALUE lines; only sets variables that are not already set."""
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))
