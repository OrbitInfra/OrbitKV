"""One documentation site, built from the repository's Markdown guides."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "_ext"))

project = "OrbitKV"
copyright = "2026, OrbitKV contributors"
extensions = ["myst_parser", "sphinxcontrib.mermaid", "legacy_urls"]
source_suffix = {".md": "markdown"}
exclude_patterns = ["_build", "tests", "requirements.*"]
myst_heading_anchors = 6
myst_fence_as_directive = ["mermaid"]
mermaid_version = "11.17.2"
html_theme = "sphinxawesome_theme"
html_title = "OrbitKV"
html_baseurl = "https://orbitinfra.github.io/OrbitKV"
html_static_path = ["_static"]
html_css_files = ["orbitkv.css"]
html_favicon = "_static/favicon.svg"
html_theme_options = {
    "logo_light": "_static/favicon.svg",
    "logo_dark": "_static/favicon.svg",
    "main_nav_links": {
        "Docs": "index",
        "Quickstart": "single-node",
        "Plan": "completion-plan",
        "GitHub": "https://github.com/OrbitInfra/OrbitKV",
    },
    "show_prev_next": True,
}
html_permalinks_icon = "<span>#</span>"
html_show_sourcelink = False
html_copy_source = False
html_use_index = False
