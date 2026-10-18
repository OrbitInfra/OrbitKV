"""CPU-only bounded metrics capture; owns no cache, transfer or GPU resources."""

import sys
import urllib.request


def main() -> None:
    port = int(sys.argv[1])
    if not 0 < port < 65536:
        raise ValueError("Invalid local metrics port")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"http://127.0.0.1:{port}/metrics", timeout=5) as response:
        raw = response.read(1024**2 + 1)
    if len(raw) > 1024**2:
        raise RuntimeError("Metrics response exceeded its byte bound")
    sys.stdout.buffer.write(raw)


if __name__ == "__main__":
    main()
