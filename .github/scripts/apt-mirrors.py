"""Copy APT sources while changing only mirror URLs; retain signature settings."""

from pathlib import Path
import re
import sys


def sources(source: Path, destination: Path, mirror: str):
    destination.mkdir(parents=True, exist_ok=True)
    replacements = [
        (
            r"https?://(?:[\w.-]+\.)?archive\.ubuntu\.com/ubuntu",
            "https://azure.archive.ubuntu.com/ubuntu"
            if mirror == "alternate"
            else "https://archive.ubuntu.com/ubuntu",
        ),
        (
            r"https?://security\.ubuntu\.com/ubuntu",
            "https://azure.archive.ubuntu.com/ubuntu"
            if mirror == "alternate"
            else "https://security.ubuntu.com/ubuntu",
        ),
        (
            r"https?://ports\.ubuntu\.com/ubuntu-ports",
            "https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports"
            if mirror == "alternate"
            else "https://ports.ubuntu.com/ubuntu-ports",
        ),
        (
            r"https?://developer\.download\.nvidia\.(?:com|cn)/",
            "https://developer.download.nvidia.cn/"
            if mirror == "alternate"
            else "https://developer.download.nvidia.com/",
        ),
    ]
    for path in [source / "sources.list", *(source / "sources.list.d").glob("*")]:
        if not path.is_file() or path.suffix not in {".list", ".sources"}:
            continue
        text = path.read_text()
        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text)
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)


if __name__ == "__main__":
    sources(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])
