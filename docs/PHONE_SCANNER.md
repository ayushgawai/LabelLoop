# Phone QR scanner → dashboard package card

An iPhone in the middle of the bench, next to the head camera, reads the parcel's QR label and the dashboard shows the package:
code, ZIP, lane (blue / red / black bin) and zone. **Display only.** The demo is scripted and WALLX already knows where the parcel
goes, so a scan never changes what the robot does.

```
iPhone Safari  ──HTTPS, one JPEG every ~0.3 s──▶  dashboard server on the laptop  ──▶  package card (http://127.0.0.1:8095)
  /scan page                                   decodes with labelloop.perception.qr (same reader as the relay)
```

The phone sends camera frames and the laptop does the decoding (zxing + the x3/unsharp retry tuned for small taped labels), so the
phone needs no app and no JS library. iOS only lets a web page use the camera over HTTPS, so the dashboard also serves an HTTPS
listener with a self-signed certificate.

## On the demo laptop (Ayush's Mac)

From the repo root, on the same Wi-Fi as the robot (`xsquarerobot-agi`):

```
.venv/bin/python -m labelloop.dashboard.server --https-port 8443
```

It prints the address for the phone, for example:

```
Dashboard at http://127.0.0.1:8095 (trials in runs/trials)
Phone scanner at https://192.168.1.157:8443/scan
```

- The HTTPS listener is open to the whole Wi-Fi, so it serves only `/scan` and `/api/package*`. Trials, camera frames and reports
  stay on the laptop-only `http://127.0.0.1:8095`.
- On the first run it creates the certificate in `runs/package/tls/` with `openssl` (macOS has it). The certificate is valid for 30
  days. Delete that folder to make a new one, for example after the laptop's IP changes.
- If macOS asks whether Python may accept incoming connections, click **Allow**. With the firewall on and no prompt, add Python in
  System Settings › Network › Firewall › Options.
- Open http://127.0.0.1:8095 on the laptop. The **package · phone QR scanner** card is at the top right.
- The board is saved in `runs/package/board.json`, so restarting the dashboard keeps it. Click **clear** on the card between runs.

## On the iPhone

1. Join the **same Wi-Fi as the laptop** (`xsquarerobot-agi`). The phone's own IP address doesn't matter, because the phone connects
   to the laptop. On mobile data or another network it can't reach the laptop.
2. In Safari, open the printed address (`https://<laptop IP>:8443/scan`).
3. "This Connection Is Not Private": tap **Show Details** › **visit this website** › **Visit Website**. You only do this once per
   certificate.
4. Tap **Start camera** › **Allow**. The status line reads `looking for a QR…`, then `read 420/…`. The bottom card shows the
   package, and the laptop's card flashes green when the package changes.
5. Mount the phone. The page asks iOS to keep the screen on. As a backup, set Settings › Display & Brightness › Auto-Lock › Never
   for the demo. Keep Safari in front: the page pauses while hidden.
6. If the label is small in view, use **2×** or **3×**. This crops the centre of the frame (shown as a dashed box) at full
   resolution, so the QR has to be inside that box.

Re-reading the same package only raises its `seen ×N` count. A different code becomes the new package, and the old one moves to the
history list.

## If the camera route fails on the day

- **Type it:** the phone page has a code box at the bottom. Typing `420/95126/001` and pressing Send updates the dashboard the same
  way.
- **From any machine on the Wi-Fi:**
  `curl -k -X POST https://<laptop IP>:8443/api/package/scan -H 'Content-Type: application/json' -d '{"code":"420/95126/001"}'`
- **Plain HTTP (an iOS Shortcut, for example):** start the dashboard with `--host 0.0.0.0` as well, then POST the code to
  `http://<laptop IP>:8095/api/package/scan`. The body can be JSON `{"code": ...}`, a form field `code=...`, or the bare text.
- "Cannot reach the laptop" on the phone: check that both are on the same Wi-Fi, that the IP matches what the dashboard printed, and
  the firewall.

## Routing shown on the card

Same rule as the relay (`labelloop.sequence.bin_for`) and the same zones as every scenario steps file:

| label | lane | zone |
|---|---|---|
| `420/940xx`, `941xx`, `950xx`, `951xx` | blue | 1, back left |
| `420/900xx`, `901xx`, `902xx` | red | 2, back right |
| anything else, or not `420/<ZIP5>/<seq>` | black | 3, right edge |

## API

| route | |
|---|---|
| `GET /scan` | the phone page |
| `GET /api/package` | `{current, history, scans, ver, phone}` (`phone.live` = a frame arrived in the last 5 s) |
| `POST /api/package/frame` | body = one JPEG → `{codes, package, new}` |
| `POST /api/package/scan` | `{"code": ...}` → `{codes, package, new}` |
| `POST /api/package/clear` | empty the board |

Code: `labelloop/dashboard/{package.py, tls.py, scan.html}`, routes in `server.py`, card in `index.html`. Tests:
`tests/test_package_scanner.py`.
