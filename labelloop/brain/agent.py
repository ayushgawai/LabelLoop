import base64
import json
import os
import urllib.request

import cv2

SYSTEM = (
    "A dual-arm robot sorts small boxes labelled with QR codes into two bins. "
    "You see three camera images: front, left_wrist, right_wrist. The user message gives the robot's "
    "current sub-goal (state_name and the instruction text being executed). "
    "Reply with JSON only: {\"done\": bool, \"target\": str, \"bin_side\": object|null, "
    "\"unsorted_boxes\": int|null, \"reason\": str}. "
    "done = true only if the current sub-goal is visibly completed in the images. "
    "target = short description of the box being worked on. "
    "In state SURVEY report unsorted_boxes: boxes on the table that are not in a bin, NOT counting boxes "
    "already set aside at the left edge of the table (those were rejected and are finished). Also report bin_side, "
    "the image side of each bin, like {\"blue\": \"left\", \"orange\": \"right\"}. Otherwise use null for those."
)


def _jpeg_url(img):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


class VisionAgent:
    def __init__(self, url, model, key="", timeout=10):
        self.url, self.model, self.key, self.timeout = url, model, key, timeout

    @classmethod
    def from_env(cls):
        url, model = os.environ.get("AGENT_CHAT_URL"), os.environ.get("AGENT_MODEL")
        if not url or not model:
            return None
        return cls(url, model, os.environ.get("AGENT_API_KEY", ""))

    def ask(self, images, state_name, context):
        """Verdict dict, or None on any failure (never raises)."""
        try:
            content = [{"type": "text", "text": json.dumps({"state_name": state_name, **context}, default=str)}]
            for name, img in images.items():
                content += [{"type": "text", "text": name},
                            {"type": "image_url", "image_url": {"url": _jpeg_url(img)}}]
            body = {"model": self.model, "response_format": {"type": "json_object"},
                    "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]}
            headers = {"Content-Type": "application/json"}
            if self.key:
                headers["Authorization"] = "Bearer " + self.key
            req = urllib.request.Request(self.url, json.dumps(body).encode(), headers=headers)
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                text = json.load(r)["choices"][0]["message"]["content"]
            return _parse(text)
        except Exception:
            return None


def _parse(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    d = json.loads(text)
    if not isinstance(d, dict):
        return None
    n = d.get("unsorted_boxes")
    return {"done": d.get("done") is True, "target": str(d.get("target") or ""),
            "bin_side": d.get("bin_side") if isinstance(d.get("bin_side"), dict) else None,
            "unsorted_boxes": n if isinstance(n, int) and not isinstance(n, bool) else None,
            "reason": str(d.get("reason") or "")}
