"""Self-contained HTML review report with embedded thumbnails."""

from __future__ import annotations

import base64
import html
import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from PIL import Image, ImageOps

from .model import Group, Photo
from .scanner import ScanResult


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def thumbnail_data_uri(path: str, max_side: int = 220) -> Optional[str]:
    try:
        with Image.open(path) as img:
            img.draft("RGB", (max_side * 2, max_side * 2))
            img = ImageOps.exif_transpose(img).convert("RGB")
            img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=78)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


_CSS = """
:root{--bg:#f6f7f9;--card:#fff;--ink:#16181d;--mute:#5d6470;--line:#e2e5ea;
--keep:#0f9d58;--keep-bg:#e6f6ee;--dupe:#c5221f;--dupe-bg:#fdecea;--accent:#3367d6}
@media (prefers-color-scheme:dark){:root{--bg:#111318;--card:#1a1d24;--ink:#e8eaee;
--mute:#9aa1ad;--line:#2a2f39;--keep:#3ddc84;--keep-bg:#13291e;--dupe:#ff8a80;--dupe-bg:#2e1716;--accent:#8ab4f8}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
header{padding:28px 24px 8px;max-width:1200px;margin:auto}
h1{margin:0 0 4px;font-size:26px;letter-spacing:-.02em}h1 span{color:var(--accent)}
.sub{color:var(--mute);margin:0 0 18px;word-break:break-all}
.stats{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:14px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 16px;min-width:140px}
.stat b{display:block;font-size:22px;letter-spacing:-.02em}.stat small{color:var(--mute)}
.bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
input[type=search]{flex:1;min-width:200px;padding:9px 12px;border-radius:10px;border:1px solid var(--line);
background:var(--card);color:var(--ink);font:inherit}
main{max-width:1200px;margin:auto;padding:8px 24px 48px}
.group{background:var(--card);border:1px solid var(--line);border-radius:14px;margin:14px 0;padding:14px 16px}
.ghead{display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap;color:var(--mute);margin-bottom:10px}
.ghead strong{color:var(--ink)}.tag{display:inline-block;padding:1px 8px;border-radius:99px;
border:1px solid var(--line);font-size:12px;margin-left:6px}
.row{display:flex;gap:12px;overflow-x:auto;padding-bottom:4px}
.photo{flex:0 0 228px;border:2px solid var(--line);border-radius:12px;overflow:hidden;background:var(--bg)}
.photo.keep{border-color:var(--keep)}.photo.dupe{border-color:var(--dupe)}
.thumb{height:170px;display:flex;align-items:center;justify-content:center;background:#0002}
.thumb img{max-width:100%;max-height:170px;display:block}
.meta{padding:8px 10px;font-size:12px;color:var(--mute)}
.meta .name{color:var(--ink);font-weight:600;font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.badge{display:inline-block;font-weight:700;font-size:11px;padding:2px 7px;border-radius:6px;margin-bottom:4px;letter-spacing:.04em}
.keep .badge{background:var(--keep-bg);color:var(--keep)}.dupe .badge{background:var(--dupe-bg);color:var(--dupe)}
a{color:var(--accent);text-decoration:none}footer{color:var(--mute);text-align:center;padding:20px}
.err{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 16px;color:var(--mute)}
"""

_JS = """
const q=document.getElementById('q');q&&q.addEventListener('input',()=>{const v=q.value.toLowerCase();
document.querySelectorAll('.group').forEach(g=>{g.style.display=g.dataset.paths.includes(v)?'':'none'})});
"""


def _photo_card(p: Photo, g: Group, thumbs: dict[str, Optional[str]], root: Path) -> str:
    keep = p is g.keeper
    try:
        rel = str(Path(p.path).relative_to(root))
    except ValueError:
        rel = p.path
    uri = Path(p.path).as_uri()
    thumb = thumbs.get(p.path)
    img = f'<img loading="lazy" src="{thumb}" alt="">' if thumb else "<small>no preview</small>"
    dist = g.distances.get(p.path, 0)
    why = "best of group" if keep else f"distance {dist}/64"
    return (
        f'<div class="photo {"keep" if keep else "dupe"}">'
        f'<a class="thumb" href="{html.escape(uri)}" target="_blank">{img}</a>'
        f'<div class="meta"><span class="badge">{"KEEP" if keep else "DUPLICATE"}</span>'
        f'<div class="name" title="{html.escape(rel)}">{html.escape(Path(p.path).name)}</div>'
        f"{p.width}&times;{p.height} &middot; {human_bytes(p.size)}<br>"
        f"sharpness {p.sharpness:,.0f} &middot; {why}<br>"
        f'<span title="{html.escape(rel)}">{html.escape(str(Path(rel).parent))}</span></div></div>'
    )


def render_report(result: ScanResult, root: Path, threshold: int, algo: str,
                  elapsed: float = 0.0) -> str:
    root = Path(root).resolve()
    needed = [p.path for g in result.groups for p in g.members]
    with ThreadPoolExecutor(max_workers=8) as pool:
        thumbs = dict(zip(needed, pool.map(thumbnail_data_uri, needed)))

    cards = []
    for n, g in enumerate(result.groups, 1):
        kind = "identical bytes" if g.identical else f"max distance {g.max_distance}"
        paths = html.escape(" ".join(p.path.lower() for p in g.members))
        body = "".join(_photo_card(p, g, thumbs, root) for p in [g.keeper, *g.duplicates])
        cards.append(
            f'<section class="group" data-paths="{paths}"><div class="ghead">'
            f"<div><strong>Group {n}</strong> &middot; {len(g.members)} photos"
            f'<span class="tag">{kind}</span></div>'
            f"<div>frees {human_bytes(g.reclaimable_bytes)}</div></div>"
            f'<div class="row">{body}</div></section>'
        )
    errors = ""
    if result.errors:
        items = "".join(
            f"<li>{html.escape(p.path)}: {html.escape(p.error or '')}</li>" for p in result.errors[:200]
        )
        errors = f'<h3>Unreadable files ({len(result.errors)})</h3><div class="err"><ul>{items}</ul></div>'
    empty = '<p class="err">No near-duplicates found. Nice and tidy.</p>' if not cards else ""
    stats = [
        (f"{len(result.photos):,}", "photos scanned"),
        (f"{len(result.groups):,}", "duplicate groups"),
        (f"{result.duplicate_count:,}", "duplicates"),
        (human_bytes(result.reclaimable_bytes), "reclaimable"),
    ]
    if elapsed:
        stats.append((f"{elapsed:.1f} s", "scan time"))
    stat_html = "".join(f'<div class="stat"><b>{v}</b><small>{k}</small></div>' for v, k in stats)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>twinpix report</title><style>{_CSS}</style></head><body>
<header><h1>twin<span>pix</span> report</h1>
<p class="sub">{html.escape(str(root))} &middot; {algo} &middot; threshold {threshold}/64</p>
<div class="stats">{stat_html}</div>
<div class="bar"><input id="q" type="search" placeholder="Filter groups by file name or folder&hellip;"></div>
</header><main>{empty}{''.join(cards)}{errors}</main>
<footer>Nothing has been moved or deleted. Run <code>twinpix move</code> to send duplicates to a review folder.</footer>
<script>{_JS}</script></body></html>"""


def write_report(result: ScanResult, root: Path, out: Path, threshold: int, algo: str,
                 elapsed: float = 0.0) -> Path:
    out = Path(out)
    out.write_text(render_report(result, root, threshold, algo, elapsed), encoding="utf-8")
    return out
