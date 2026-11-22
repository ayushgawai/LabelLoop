"""HTTPS for the phone scanner: iOS Safari only gives a page the camera over HTTPS, so the dashboard can serve a second, TLS listener
with a self-signed certificate (made once with the `openssl` CLI: macOS ships one, Git for Windows too). The phone shows a
"not private" warning once; Show Details -> visit this website.
"""
import shutil
import socket
import ssl
import subprocess
import tempfile
from http.server import ThreadingHTTPServer
from pathlib import Path

HANDSHAKE_TIMEOUT_S = 15


def lan_ips():
    """This machine's IPv4 addresses that a phone on the same Wi-Fi could reach (best effort, no traffic sent)."""
    ips = []
    for probe in ("192.168.1.1", "10.255.255.255", "8.8.8.8"):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect((probe, 80))  # UDP connect: picks the outgoing interface, sends nothing
            ip = s.getsockname()[0]
            if ip not in ips and not ip.startswith("127."):
                ips.append(ip)
        except OSError:
            pass
        finally:
            s.close()
    try:
        for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
            if ip not in ips and not ip.startswith("127."):
                ips.append(ip)
    except OSError:
        pass
    return ips


def ensure_cert(folder, ips=None, openssl=None):
    """(cert.pem, key.pem) in `folder`, made with openssl if missing. Valid 30 days, SAN = the given IPs + localhost."""
    folder = Path(folder)
    cert, key = folder / "cert.pem", folder / "key.pem"
    if cert.is_file() and key.is_file():
        return cert, key
    exe = openssl or shutil.which("openssl")
    if not exe:
        raise RuntimeError("no `openssl` on PATH: install it or pass --cert/--key for the HTTPS scanner")
    folder.mkdir(parents=True, exist_ok=True)
    san = ["DNS:localhost", "IP:127.0.0.1"] + ["IP:%s" % ip for ip in (ips if ips is not None else lan_ips())]
    cnf = ("[req]\ndistinguished_name = dn\nprompt = no\n[dn]\nCN = LabelLoop scanner\n"
           "[v3]\nbasicConstraints = CA:FALSE\nkeyUsage = digitalSignature, keyEncipherment\n"
           "extendedKeyUsage = serverAuth\nsubjectAltName = %s\n" % ", ".join(san))
    with tempfile.TemporaryDirectory() as tmp:  # an explicit config: Git for Windows' openssl has no default one
        cfg = Path(tmp) / "req.cnf"
        cfg.write_text(cnf, encoding="ascii")
        subprocess.run([exe, "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-sha256", "-days", "30",
                        "-keyout", str(key), "-out", str(cert), "-config", str(cfg), "-extensions", "v3"],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return cert, key


class TLSServer(ThreadingHTTPServer):
    """ThreadingHTTPServer over TLS. The handshake runs in the request's own thread (with a timeout), so a phone that stalls on the
    certificate warning never blocks the accept loop."""

    def __init__(self, addr, handler, cert, key):
        self.ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        self.ctx.load_cert_chain(str(cert), str(key))
        super().__init__(addr, handler)

    def get_request(self):
        sock, addr = self.socket.accept()
        sock.settimeout(HANDSHAKE_TIMEOUT_S)
        return self.ctx.wrap_socket(sock, server_side=True, do_handshake_on_connect=False), addr

    def finish_request(self, request, client_address):
        try:
            request.do_handshake()
        except (ssl.SSLError, OSError):  # the phone said no to the certificate, or went away: nothing to serve
            return
        super().finish_request(request, client_address)

    def handle_error(self, request, client_address):  # quiet, like the handler: the relay owns stdout/stderr
        pass
