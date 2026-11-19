import json
import os
import time


def _default(o):
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


class RunLog:
    """One JSON line per write(). path=None is a no-op."""

    def __init__(self, path):
        self.path = path
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def write(self, **fields):
        if self.path:
            with open(self.path, "a") as f:
                f.write(json.dumps({"t": time.time(), **fields}, default=_default) + "\n")
