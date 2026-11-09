"""Preserve published Astro URLs while Sphinx owns navigation and search."""

import html
import json
import posixpath
import shutil
from pathlib import Path

from docutils import nodes
from sphinx.errors import ExtensionError


def redirects(source: Path) -> dict[str, str]:
    """The old /docs/<slug>/ URLs and aliases retired before this migration."""
    return {
        **{f"docs/{p.stem}": p.stem for p in source.glob("*.md") if p.stem != "index"},
        **json.loads((source / "redirects.json").read_text()),
    }


def preserve_fragments(app, doctree, docname):
    """Keep Markdown/GitHub heading slugs as well as Sphinx's normalized IDs."""
    by_id = {
        key: node for node in doctree.findall(nodes.Element) for key in node["ids"]
    }
    for slug, (_, target, _) in app.env.metadata[docname].get("myst_slugs", {}).items():
        if slug not in by_id and target in by_id:
            by_id[target]["ids"].append(slug)


def write_legacy_urls(app, exception):
    if exception is not None or app.builder.name != "dirhtml":
        return
    source, output = Path(app.srcdir), Path(app.outdir)
    base = app.config.html_baseurl.rstrip("/") + "/"
    for old, new in redirects(source).items():
        if new not in app.env.found_docs:
            raise ExtensionError(f"Redirect {old}: unknown document {new}")
        uri = app.builder.get_target_uri(new)
        target = posixpath.relpath(uri or ".", old) + "/"
        destination = output / old / "index.html"
        if old in app.env.found_docs:
            raise ExtensionError(f"Redirect would overwrite {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            "<title>Page moved · OrbitKV</title>"
            f'<link rel="canonical" href="{html.escape(base + uri)}">'
            # Script precedes refresh so query strings and fragments survive.
            f"<script>location.replace({json.dumps(target)} + location.search + location.hash);</script>"
            f'<meta http-equiv="refresh" content="0; url={html.escape(target)}">'
            f'</head><body><p>Page moved to <a href="{html.escape(target)}">'
            "the OrbitKV documentation</a>.</p></body></html>\n"
        )
    for old, new in {
        "architecture.svg": "architecture-light.svg",
        "readme-banner.svg": "logo-light.svg",
        "favicon.svg": "favicon.svg",
    }.items():
        shutil.copyfile(source / "_static" / new, output / old)
    # Cached copies of the old search UI can still fetch their document index.
    (output / "docs" / "search.json").write_text(
        json.dumps(
            [
                {
                    "title": app.env.titles[p.stem].astext(),
                    "url": base + app.builder.get_target_uri(p.stem),
                    "body": p.read_text(),
                }
                for p in sorted(source.glob("*.md"))
                if p.stem != "index"
            ]
        )
        + "\n"
    )
    (output / "404.html").write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>Page not found · OrbitKV</title></head><body><h1>Page not found</h1>"
        f'<p><a href="{base}">Browse the documentation</a> or '
        f'<a href="{base}search/">search OrbitKV</a>.</p></body></html>\n'
    )
    (output / ".nojekyll").touch()


def setup(app):
    app.connect("doctree-resolved", preserve_fragments)
    app.connect("build-finished", write_legacy_urls)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
