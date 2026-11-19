import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from labelloop.brain.agent import VisionAgent

IMGS = {"front": np.zeros((32, 32, 3), np.uint8)}


@pytest.fixture
def server():
    box = {"reply": "", "status": 200, "req": None}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            box["req"] = (json.loads(self.rfile.read(int(self.headers["Content-Length"]))), self.headers["Authorization"])
            self.send_response(box["status"])
            self.end_headers()
            self.wfile.write(json.dumps({"choices": [{"message": {"content": box["reply"]}}]}).encode())

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    box["url"] = f"http://127.0.0.1:{srv.server_port}/v1/chat/completions"
    yield box
    srv.shutdown()


def test_parses_fenced_json_and_sends_images(server):
    server["reply"] = '```json\n{"done": true, "target": "box", "bin_side": {"blue": "left"}, "unsorted_boxes": 2, "reason": "ok"}\n```'
    out = VisionAgent(server["url"], "m", "k").ask(IMGS, "SURVEY", {"instruction": "x"})
    assert out == {"done": True, "target": "box", "bin_side": {"blue": "left"}, "unsorted_boxes": 2, "reason": "ok"}
    body, auth = server["req"]
    assert auth == "Bearer k" and body["response_format"] == {"type": "json_object"}
    assert body["messages"][1]["content"][2]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_missing_fields_default(server):
    server["reply"] = '{"done": "yes"}'
    assert VisionAgent(server["url"], "m").ask(IMGS, "LOOK", {}) == {
        "done": False, "target": "", "bin_side": None, "unsorted_boxes": None, "reason": ""}


@pytest.mark.parametrize("reply,status", [("not json", 200), ("[1]", 200), ("{}", 500)])
def test_errors_return_none(server, reply, status):
    server["reply"], server["status"] = reply, status
    assert VisionAgent(server["url"], "m").ask(IMGS, "LOOK", {}) is None


def test_unreachable_returns_none():
    assert VisionAgent("http://127.0.0.1:9/x", "m", timeout=1).ask(IMGS, "LOOK", {}) is None


def test_from_env(monkeypatch):
    monkeypatch.delenv("AGENT_CHAT_URL", raising=False)
    assert VisionAgent.from_env() is None
    monkeypatch.setenv("AGENT_CHAT_URL", "http://x")
    monkeypatch.setenv("AGENT_MODEL", "m")
    assert VisionAgent.from_env().model == "m"
