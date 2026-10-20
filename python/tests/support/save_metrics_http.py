"""CPU-only bounded metrics capture; owns no cache, transfer or GPU resources."""

import sys
import urllib.request


class FixedOrigin(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, url):
        raise RuntimeError("Metrics endpoint redirect is forbidden")


def main() -> None:
    port = int(sys.argv[1])
    if not 0 < port < 65536:
        raise ValueError("Invalid local metrics port")
    url = f"http://127.0.0.1:{port}/metrics"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), FixedOrigin())
    with opener.open(url, timeout=5) as response:
        if response.status != 200 or response.geturl() != url:
            raise RuntimeError("Metrics endpoint identity or status changed")
        raw = response.read(1024**2 + 1)
    if len(raw) > 1024**2:
        raise RuntimeError("Metrics response exceeded its byte bound")
    sys.stdout.buffer.write(raw)


if __name__ == "__main__":
    main()
