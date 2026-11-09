"""Check the built Pages site, including fragments and historical source links."""

import argparse
import json
import re
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

from _ext.legacy_urls import redirects

BASE = "https://orbitinfra.github.io/OrbitKV/"
REPOSITORY = Path(__file__).resolve().parent.parent


class Page(HTMLParser):
    def __init__(self, text: str):
        super().__init__(convert_charrefs=True)
        self.ids: set[str] = set()
        self.links: list[str] = []
        self.redirect: str | None = None
        self.feed(text)

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        for key in ("href", "src", "action"):
            if attrs.get(key):
                self.links.append(attrs[key])
        if tag == "meta" and attrs.get("http-equiv", "").lower() == "refresh":
            self.redirect = attrs["content"].split("url=", 1)[1].strip()
            self.links.append(self.redirect)


def check_links(output: Path, repository: Path = REPOSITORY) -> list[str]:
    output = output.resolve()
    pages = {p: Page(p.read_text()) for p in output.rglob("*.html")}
    errors = []
    checked_sources = set()

    def check(url: str, seen: frozenset[str] = frozenset()):
        target = urlsplit(url)
        path = unquote(target.path)
        if target.netloc == "github.com":
            match = re.fullmatch(
                r"/OrbitInfra/OrbitKV/(?:blob|tree)/([^/]+)/(.*)", path
            )
            if match and path not in checked_sources:
                checked_sources.add(path)
                revision, source = match.groups()
                if re.fullmatch(r"[0-9a-f]{40}", revision):
                    result = subprocess.run(
                        ["git", "cat-file", "-e", f"{revision}:{source}"],
                        cwd=repository,
                        capture_output=True,
                        check=False,
                    )
                    if result.returncode:
                        errors.append(
                            f"Missing historical source {url}; fetch full Git history"
                        )
                elif not (repository / source).exists():
                    errors.append(f"Missing repository source {url}")
            return
        if target.netloc != urlsplit(BASE).netloc or target.scheme not in {
            "http",
            "https",
        }:
            return
        if not path.startswith("/OrbitKV/"):
            errors.append(f"URL outside Pages base: {url}")
            return
        destination = (output / path.removeprefix("/OrbitKV/")).resolve()
        if not destination.is_relative_to(output):
            errors.append(f"URL escapes build: {url}")
            return
        if destination.is_dir():
            destination /= "index.html"
        if not destination.is_file():
            errors.append(f"Missing published target: {url}")
            return
        if destination in pages:
            page = pages[destination]
            if page.redirect:
                if url in seen:
                    errors.append(f"Redirect loop: {url}")
                else:
                    redirect = urljoin(url, page.redirect)
                    if target.fragment:
                        redirect += "#" + target.fragment
                    check(redirect, seen | {url})
            elif target.fragment and unquote(target.fragment) not in page.ids:
                errors.append(f"Missing fragment: {url}")

    if not pages:
        errors.append("No built HTML; run sphinx-build first")
    for path, page in pages.items():
        current = urljoin(
            BASE, path.relative_to(output).as_posix().removesuffix("index.html")
        )
        for link in page.links:
            check(urljoin(current, link))
    return errors


def check_site(output: Path, source: Path) -> list[str]:
    errors = check_links(output)
    expected = {p.stem for p in source.glob("*.md")}
    search = output / "searchindex.js"
    if not search.exists():
        errors.append("Missing Sphinx search index")
    else:
        index = json.loads(
            search.read_text()
            .strip()
            .removeprefix("Search.setIndex(")
            .removesuffix(";")
            .removesuffix(")")
        )
        if set(index["docnames"]) != expected:
            errors.append("Search index does not cover every guide exactly once")
    for old, new in redirects(source).items():
        path = output / old / "index.html"
        if not path.exists():
            errors.append(f"Missing legacy URL: {old}")
            continue
        page = Page(path.read_text())
        target = BASE + (new + "/" if new != "index" else "")
        if not page.redirect or urljoin(BASE + old + "/", page.redirect) != target:
            errors.append(f"Incorrect legacy redirect: {old} → {new}")
        if "location.search + location.hash" not in path.read_text():
            errors.append(f"Legacy redirect loses query/fragment: {old}")
    for name in expected:
        if not (output / (name if name != "index" else "") / "index.html").exists():
            errors.append(f"Missing canonical guide: {name}")
    return errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    failures = check_site(args.output, Path(__file__).parent)
    if failures:
        parser.exit(1, "\n".join(sorted(set(failures))) + "\n")
    print(
        "Published links, fragments, source targets, redirects and search coverage passed"
    )
