"""CPU checks for the bounded Save-byte measurement used by GPU tests."""

import subprocess
import threading
import time

import pytest


def capture_save_metrics(monkeypatch, raw):
    from tests.integration import test_pending_save_bytes as byte_gate

    monkeypatch.setattr(
        byte_gate.subprocess,
        "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0, stdout=raw, stderr=b""),
    )


@pytest.mark.parametrize(
    "text",
    [
        "",
        "NaN",
        "inf",
        "-1",
        "0.5",
        "9007199254740994",
        "9007199254740993",
        "9007199254740991.5",
        "1_6384",
    ],
)
def test_save_byte_snapshot_rejects_missing_or_inexact_counter(monkeypatch, text):
    from tests.integration.test_pending_save_bytes import save_bytes_snapshot

    capture_save_metrics(monkeypatch, f"orbitkv_save_bytes_total {text}\n".encode())
    with pytest.raises(RuntimeError, match="counter is missing"):
        save_bytes_snapshot(1)


@pytest.mark.parametrize("text", ["0", "16384.0", "1.6384e4", "9007199254740992"])
def test_save_byte_snapshot_preserves_exact_raw_integer(monkeypatch, text):
    from decimal import Decimal

    from tests.integration.test_pending_save_bytes import save_bytes_snapshot

    capture_save_metrics(monkeypatch, f"orbitkv_save_bytes_total {text}\n".encode())
    value = save_bytes_snapshot(1)
    assert type(value) is int and value == int(Decimal(text))


@pytest.mark.parametrize(
    "body",
    [
        "orbitkv_save_bytes_total 0\n" * 2,
        'orbitkv_save_bytes_total{label="' + "x" * 1024 + '"} 16384\n',
        "# " + "x" * 4096 + "\norbitkv_save_bytes_total 0\n",
        "# x\n" * (1024**2 // 4 + 1) + "orbitkv_save_bytes_total 0\n",
    ],
    ids=["duplicate", "oversized-label", "long-line", "large-body"],
)
def test_save_byte_snapshot_bounds_raw_response(monkeypatch, body):
    from tests.integration.test_pending_save_bytes import save_bytes_snapshot

    capture_save_metrics(monkeypatch, body.encode())
    with pytest.raises(RuntimeError, match="counter"):
        save_bytes_snapshot(1)


def test_save_byte_snapshot_consumes_actual_exporter_labels(monkeypatch):
    from tests.integration.test_pending_save_bytes import save_bytes_snapshot

    raw = (
        b'orbitkv_save_bytes_total{otel_scope_name="orbitkv-core",otel_scope_schema_url=""} 8192\n'
        b'orbitkv_save_bytes_total{otel_scope_name="other-core",otel_scope_schema_url=""} 8192.0\n'
    )
    capture_save_metrics(monkeypatch, raw)
    assert save_bytes_snapshot(1) == 16384


@pytest.mark.integration
@pytest.mark.parametrize("mode", ["normal", "oversized", "trickle", "slow-headers", "redirect"])
def test_save_byte_snapshot_actual_http_is_bounded(mode):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from tests.integration.test_pending_save_bytes import save_bytes_snapshot

    stopped = threading.Event()
    foreign_requests = []

    class Trickle(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/metrics":
                foreign_requests.append(self.path)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"orbitkv_save_bytes_total 16384\n")
                return
            if mode == "redirect":
                self.send_response(302)
                self.send_header("Location", "/other-metrics")
                self.end_headers()
                return
            if mode == "slow-headers":
                self.connection.settimeout(1)
                self.wfile.write(b"HTTP/1.1 200 ")
                self.wfile.flush()
                while not stopped.wait(4):
                    try:
                        self.wfile.write(b"x")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        return
                return
            self.send_response(200)
            if mode in ("normal", "oversized"):
                body = b"orbitkv_save_bytes_total 0\n"
                if mode == "oversized":
                    body = b"# x\n" * (1024**2 // 4 + 1)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                self.wfile.flush()
                return
            self.send_header("Content-Length", "100")
            self.end_headers()
            while not stopped.wait(4):
                try:
                    self.wfile.write(b"x")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Trickle)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        started = time.monotonic()
        if mode == "normal":
            assert save_bytes_snapshot(server.server_port) == 0
        else:
            with pytest.raises(RuntimeError, match="time bound|byte bound|response failed"):
                save_bytes_snapshot(server.server_port)
        assert time.monotonic() - started < 11
        assert not foreign_requests
    finally:
        stopped.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive()
