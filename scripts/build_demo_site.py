"""Build the GitHub Pages demo: a real twinpix report of the synthetic library.

    python scripts/build_demo_site.py

Generates the demo library in a temporary folder, scans it, and writes the
report to docs/index.html (with links to the original files turned off,
since those only exist on the machine that ran the scan).
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from twinpix import demo
from twinpix.report import write_report
from twinpix.scanner import scan

DOCS = Path(__file__).resolve().parents[1] / "docs"


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "Phone Backup"
        demo.build(root, scenes=120, seed=7)
        (root / "truth.json").unlink()
        t0 = time.perf_counter()
        result = scan(root)
        elapsed = time.perf_counter() - t0
        out = write_report(result, root, DOCS / "index.html", threshold=10, algo="phash",
                           elapsed=elapsed, link_originals=False)
        html = out.read_text(encoding="utf-8").replace(str(root.resolve()), "demo library (synthetic)")
        banner = ('<div style="background:#3367d6;color:#fff;text-align:center;padding:8px 16px;'
                  'font-size:13px">Live demo: a real twinpix report of a synthetic 316-photo library. '
                  '<a style="color:#fff;text-decoration:underline" '
                  'href="https://github.com/aadijewargi/twinpix">Get twinpix on GitHub</a></div>')
        html = html.replace("<body>", "<body>" + banner, 1)
        out.write_text(html, encoding="utf-8")
    print(f"{len(result.photos)} photos, {len(result.groups)} groups -> {out}")


if __name__ == "__main__":
    main()
