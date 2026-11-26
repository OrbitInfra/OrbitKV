"""An isolated etcd process shared by distributed runtime gates."""

import contextlib
import errno
import os
import secrets
import select
import socket
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import urlopen


def find_available_port() -> int:
    """Find a listener port outside outgoing TCP and OrbiTE listener-port ranges."""
    low, high = map(int, Path("/proc/sys/net/ipv4/ip_local_port_range").read_text().split())
    for _ in range(128):
        port = 1024 + secrets.randbelow(65536 - 1024)
        # OrbiTE listener starts before the engine HTTP listener and
        # independently chooses an RPC port in [15000, 17000).
        if low <= port <= high or 15000 <= port < 17000:
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                s.listen(1)
            except OSError as error:
                if error.errno != errno.EADDRINUSE:
                    raise
                continue
            return s.getsockname()[1]
    raise RuntimeError("No free test listener port found")


class TcpGate:
    """Test-owned TCP fault gate that can sever and reject existing streams."""

    def __init__(self, target: str):
        parsed = urlsplit(target)
        if parsed.scheme != "http" or parsed.hostname is None or parsed.port is None:
            raise ValueError(f"unsupported TCP gate target: {target}")
        self._target = (parsed.hostname, parsed.port)
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind(("127.0.0.1", find_available_port()))
        self._listener.listen()
        self._listener.settimeout(0.1)
        self.endpoint = f"http://127.0.0.1:{self._listener.getsockname()[1]}"
        self._condition = threading.Condition()
        self._connections: set[tuple[socket.socket, socket.socket]] = set()
        self._partitioned = False
        self._drop_responses = False
        self._dropped_response_bytes = 0
        self._delay_seconds = 0.0
        self._stopped = False
        self._thread = threading.Thread(target=self._serve, name="orbitkv-etcd-gate", daemon=True)
        self._thread.start()

    def partition(self) -> None:
        with self._condition:
            self._partitioned = True
            connections = list(self._connections)
        for pair in connections:
            self._close_pair(pair)
        deadline = time.monotonic() + 5
        with self._condition:
            while self._connections:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("TCP gate connections did not drain")
                self._condition.wait(remaining)

    def drop_responses(self) -> None:
        """Discard traffic from the target while continuing to forward requests."""
        with self._condition:
            self._dropped_response_bytes = 0
            self._drop_responses = True

    def wait_for_dropped_response(self) -> int:
        with self._condition:
            if not self._condition.wait_for(lambda: self._dropped_response_bytes > 0, timeout=5):
                raise TimeoutError("TCP gate did not observe a response to drop")
            return self._dropped_response_bytes

    def heal(self) -> None:
        with self._condition:
            self._partitioned = False
            self._drop_responses = False

    def set_delay(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("TCP gate delay cannot be negative")
        with self._condition:
            self._delay_seconds = seconds

    def close(self) -> None:
        with self._condition:
            self._stopped = True
            connections = list(self._connections)
        with contextlib.suppress(OSError):
            self._listener.close()
        for pair in connections:
            self._close_pair(pair)
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            raise TimeoutError("TCP gate listener did not stop")

    def _serve(self) -> None:
        while True:
            with self._condition:
                if self._stopped:
                    return
            try:
                downstream, _ = self._listener.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            with self._condition:
                if self._partitioned or self._stopped:
                    downstream.close()
                    continue
            try:
                upstream = socket.create_connection(self._target, timeout=2)
                downstream.settimeout(None)
                upstream.settimeout(None)
            except OSError:
                downstream.close()
                continue
            pair = (downstream, upstream)
            with self._condition:
                if self._partitioned or self._stopped:
                    self._close_pair(pair)
                    continue
                self._connections.add(pair)
            threading.Thread(
                target=self._proxy,
                args=(pair,),
                name="orbitkv-etcd-gate-connection",
                daemon=True,
            ).start()

    def _proxy(self, pair: tuple[socket.socket, socket.socket]) -> None:
        downstream, upstream = pair
        try:
            while True:
                with self._condition:
                    if self._partitioned or self._stopped:
                        return
                readable, _, _ = select.select(pair, (), (), 0.1)
                for source in readable:
                    destination = upstream if source is downstream else downstream
                    data = source.recv(16 * 1024)
                    if not data:
                        return
                    with self._condition:
                        if source is upstream and self._drop_responses:
                            self._dropped_response_bytes += len(data)
                            self._condition.notify_all()
                            continue
                        delay = self._delay_seconds
                    if delay:
                        time.sleep(delay)
                    destination.sendall(data)
        except (OSError, ValueError):
            pass
        finally:
            self._close_pair(pair)
            with self._condition:
                self._connections.discard(pair)
                self._condition.notify_all()

    @staticmethod
    def _close_pair(pair: tuple[socket.socket, socket.socket]) -> None:
        for connection in pair:
            with contextlib.suppress(OSError):
                connection.shutdown(socket.SHUT_RDWR)
            with contextlib.suppress(OSError):
                connection.close()


@contextmanager
def etcd_server(directory):
    binary = os.environ.get("ETCD_BIN")
    if not binary:
        import pytest

        pytest.skip("set ETCD_BIN to run a distributed runtime gate")
    endpoint = f"http://127.0.0.1:{find_available_port()}"
    peer = f"http://127.0.0.1:{find_available_port()}"
    log_path = directory / "etcd.log"
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            [
                binary,
                "--name",
                "test",
                "--data-dir",
                str(directory / "etcd-data"),
                "--listen-client-urls",
                endpoint,
                "--advertise-client-urls",
                endpoint,
                "--listen-peer-urls",
                peer,
                "--initial-advertise-peer-urls",
                peer,
                "--initial-cluster",
                f"test={peer}",
                "--log-level",
                "warn",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )

    def stop():
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    try:
        deadline = time.monotonic() + 15
        while True:
            assert process.poll() is None, log_path.read_text()
            try:
                with urlopen(f"{endpoint}/health", timeout=1) as response:
                    if response.status == 200:
                        break
            except (URLError, TimeoutError, ConnectionError):
                pass
            assert time.monotonic() < deadline, "etcd startup timed out"
            time.sleep(0.05)
        yield endpoint, stop
    finally:
        stop()
