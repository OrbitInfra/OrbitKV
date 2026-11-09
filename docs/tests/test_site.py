"""Negative controls for the publication gate; no server or network required."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_site import check_links  # noqa: E402


class PublicationTests(unittest.TestCase):
    def test_links_and_fragments_include_redirect_destinations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "guide").mkdir()
            (root / "old").mkdir()
            (root / "index.html").write_text('<a href="old/#ready">Guide</a>')
            redirect = root / "old/index.html"
            redirect.write_text(
                '<meta http-equiv="refresh" content="0; url=../guide/">'
            )
            target = root / "guide/index.html"
            target.write_text('<h1 id="ready">Ready</h1>')
            self.assertEqual(check_links(root), [])
            target.write_text('<h1 id="drained">Drained</h1>')
            self.assertTrue(
                any("Missing fragment" in error for error in check_links(root))
            )
            target.unlink()
            self.assertTrue(
                any("Missing published target" in error for error in check_links(root))
            )

    def test_invalid_publication_paths_fail(self):
        cases = {
            "asset": ('<img src="missing.svg">', "Missing published target"),
            "base": ('<a href="/guide/">Guide</a>', "outside Pages base"),
            "escape": ('<a href="/OrbitKV/%2e%2e/secret">Bad</a>', "escapes build"),
            "source": (
                '<a href="https://github.com/OrbitInfra/OrbitKV/blob/main/missing.rs">Source</a>',
                "Missing repository source",
            ),
            "loop": (
                '<meta http-equiv="refresh" content="0; url=./">',
                "Redirect loop",
            ),
        }
        for name, (html, message) in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "index.html").write_text(html)
                self.assertTrue(
                    any(message in error for error in check_links(root, root))
                )


if __name__ == "__main__":
    unittest.main()
