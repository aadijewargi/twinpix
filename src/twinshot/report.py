"""Self-contained HTML review report (thumbnails embedded, no server needed).

The report is a single file: open it in any browser, click photos to flip them
between KEEP and MOVE, then press "Download plan" and feed the JSON to
``twinshot apply``.
"""

from __future__ import annotations

import base64
import html
import io
import json
import os
from typing import Optional, Sequence

from .grouping import ScanResult
from .hashing import open_image

THUMB_SIDE = 280


def _thumbnail_data_uri(path: str, side: int = THUMB_SIDE) -> Optional[str]:
    try:
        img, _ = open_image(path, draft_size=(side * 2, side * 2))
        img.thumbnail((side, side))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=78, optimize=True)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


def _display_path(path: str, roots: Sequence[str]) -> str:
    for r in roots:
        r = os.path.abspath(r)
        try:
            if os.path.commonpath([os.path.abspath(path), r]) == r:
                return os.path.relpath(path, r).replace("\\", "/")
        except ValueError:
            continue
    return path


def report_data(result: ScanResult, plan: dict, roots: Sequence[str]) -> dict:
    groups = []
    for g in result.groups:
        members = []
        for m in g.members:
            f = m.fp
            members.append(
                {
                    "path": f.path,
                    "label": _display_path(f.path, roots),
                    "w": f.width,
                    "h": f.height,
                    "bytes": f.bytes,
                    "sharp": round(f.sharpness, 1),
                    "pd": m.phash_dist,
                    "dd": m.dhash_dist,
                    "exact": m.exact_copy,
                    "thumb": _thumbnail_data_uri(f.path),
                }
            )
        groups.append({"id": g.id, "kind": g.kind, "reason": g.reason, "members": members})
    return {
        "scanned": result.scanned,
        "skipped": len(result.skipped),
        "roots": [os.path.abspath(r) for r in roots],
        "groups": groups,
        "plan": plan,
    }


def render_report(result: ScanResult, plan: dict, roots: Sequence[str], title: str = "twinshot report") -> str:
    data = report_data(result, plan, roots)
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return (
        _TEMPLATE.replace("__TITLE__", html.escape(title))
        .replace("__DATA__", payload)
    )


def write_report(result: ScanResult, plan: dict, roots: Sequence[str], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render_report(result, plan, roots))


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
  --bg: #f6f5f2; --panel: #ffffff; --ink: #1d1d1f; --muted: #6b6b70; --line: #e4e2dc;
  --keep: #1f8a4c; --keep-bg: #e5f4ea; --move: #c2410c; --move-bg: #fdece3;
  --accent: #3b5bdb; --warn: #b42318; --shadow: 0 1px 2px rgba(0,0,0,.06), 0 4px 16px rgba(0,0,0,.05);
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #121214; --panel: #1c1c1f; --ink: #ececee; --muted: #9a9aa2; --line: #2c2c31;
    --keep: #4ade80; --keep-bg: #13301f; --move: #fb923c; --move-bg: #3a2113;
    --accent: #7c95ff; --warn: #f87171; --shadow: none;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
header { position: sticky; top: 0; z-index: 5; background: color-mix(in srgb, var(--bg) 88%, transparent);
  backdrop-filter: blur(8px); border-bottom: 1px solid var(--line); }
.bar { max-width: 1240px; margin: 0 auto; padding: 14px 20px; display: flex; flex-wrap: wrap; gap: 14px 24px; align-items: center; }
.brand { font-weight: 700; font-size: 18px; letter-spacing: -.01em; }
.brand span { color: var(--muted); font-weight: 500; font-size: 13px; margin-left: 6px; }
.stats { display: flex; gap: 20px; flex-wrap: wrap; }
.stat b { display: block; font-size: 18px; font-variant-numeric: tabular-nums; }
.stat small { color: var(--muted); }
.actions { margin-left: auto; display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
button, select { font: inherit; border-radius: 8px; border: 1px solid var(--line); background: var(--panel);
  color: var(--ink); padding: 7px 12px; cursor: pointer; }
button.primary { background: var(--accent); border-color: var(--accent); color: #fff; font-weight: 600; }
button:disabled { opacity: .45; cursor: not-allowed; }
main { max-width: 1240px; margin: 0 auto; padding: 20px; }
.hint { color: var(--muted); margin: 0 0 16px; }
.group { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 14px 16px 16px;
  margin-bottom: 16px; box-shadow: var(--shadow); }
.ghead { display: flex; gap: 10px; align-items: baseline; flex-wrap: wrap; margin-bottom: 12px; }
.ghead h2 { font-size: 15px; margin: 0; }
.chip { font-size: 12px; padding: 2px 8px; border-radius: 99px; border: 1px solid var(--line); color: var(--muted); }
.chip.exact { color: var(--accent); border-color: currentColor; }
.ghead .why { color: var(--muted); font-size: 13px; }
.ghead .save { margin-left: auto; color: var(--muted); font-size: 13px; font-variant-numeric: tabular-nums; }
.warn { color: var(--warn); font-weight: 600; font-size: 13px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); gap: 12px; }
.tile { border: 2px solid var(--line); border-radius: 11px; overflow: hidden; cursor: pointer; position: relative;
  background: var(--panel); transition: border-color .12s, transform .12s; user-select: none; }
.tile:hover { transform: translateY(-1px); }
.tile.keep { border-color: var(--keep); }
.tile.move { border-color: var(--move); }
.tile.move .img img { opacity: .55; filter: grayscale(.5); }
.img { aspect-ratio: 4/3; background: var(--bg); display: grid; place-items: center; overflow: hidden; }
.img img { width: 100%; height: 100%; object-fit: cover; display: block; }
.badge { position: absolute; z-index: 2; top: 8px; left: 8px; font-size: 11px; font-weight: 700; letter-spacing: .04em;
  padding: 3px 8px; border-radius: 6px; }
.keep .badge { background: var(--keep-bg); color: var(--keep); }
.move .badge { background: var(--move-bg); color: var(--move); }
.meta { padding: 8px 10px 10px; font-size: 12px; }
.name { font-weight: 600; font-size: 13px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.dir { color: var(--muted); font-size: 11.5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-bottom: 3px; }
.row { display: flex; justify-content: space-between; gap: 6px; white-space: nowrap; color: var(--muted); font-variant-numeric: tabular-nums; }
.tag { color: var(--accent); font-weight: 600; }
footer { max-width: 1240px; margin: 0 auto; padding: 8px 20px 40px; color: var(--muted); }
code { background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 1px 6px; }
.empty { text-align: center; padding: 60px 0; color: var(--muted); font-size: 16px; }
@media (max-width: 640px) { .bar { padding: 12px 16px; } main { padding: 16px; } .actions { margin-left: 0; }
  .tiles { grid-template-columns: repeat(2, 1fr); gap: 8px; } .meta { font-size: 11px; padding: 7px 8px 8px; } }
</style>
</head>
<body>
<header><div class="bar">
  <div class="brand">twinshot<span id="sub"></span></div>
  <div class="stats">
    <div class="stat"><b id="s-groups">0</b><small>groups</small></div>
    <div class="stat"><b id="s-move">0</b><small>to quarantine</small></div>
    <div class="stat"><b id="s-bytes">0</b><small>reclaimable</small></div>
  </div>
  <div class="actions">
    <select id="filter" aria-label="Filter groups">
      <option value="all">All groups</option>
      <option value="exact">Exact copies</option>
      <option value="similar">Similar shots</option>
    </select>
    <button id="reset" title="Restore the suggested keepers">Reset</button>
    <button id="download" class="primary">Download plan</button>
  </div>
</div></header>
<main>
  <p class="hint">Click a photo to flip it between <b style="color:var(--keep)">KEEP</b> and <b style="color:var(--move)">MOVE</b>.
  Nothing is deleted: moved photos go to a quarantine folder you can restore with <code>twinshot undo</code>.</p>
  <div id="groups"></div>
</main>
<footer>Then run <code>twinshot apply twinshot-plan.json</code> &middot; <span id="foot"></span></footer>
<script id="data" type="application/json">__DATA__</script>
<script>
(() => {
  const DATA = JSON.parse(document.getElementById('data').textContent);
  const state = new Map(); // path -> 'keep' | 'move'
  const suggested = () => {
    for (const g of DATA.plan.groups) {
      g.keep.forEach(p => state.set(p, 'keep'));
      g.move.forEach(p => state.set(p, 'move'));
    }
  };
  suggested();
  const fmtBytes = b => {
    const u = ['B','KB','MB','GB','TB']; let i = 0;
    while (b >= 1024 && i < u.length - 1) { b /= 1024; i++; }
    return (i ? b.toFixed(b < 10 ? 1 : 0) : b) + ' ' + u[i];
  };
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
  const root = document.getElementById('groups');
  const sub = document.getElementById('sub');
  sub.textContent = DATA.roots.map(r => r.split(/[\\/]/).filter(Boolean).pop() || r).join(', ');
  sub.title = DATA.roots.join('\n');
  document.getElementById('foot').textContent =
    `${DATA.scanned} photos scanned` + (DATA.skipped ? `, ${DATA.skipped} unreadable files skipped` : '');

  function render() {
    const filter = document.getElementById('filter').value;
    root.replaceChildren();
    const shown = DATA.groups.filter(g => filter === 'all' || g.kind === filter);
    if (!shown.length) { root.append(el('div', 'empty', DATA.groups.length ? 'No groups match this filter.' : 'No duplicates found. Your library is clean.')); }
    for (const g of shown) {
      const card = el('section', 'group');
      const head = el('div', 'ghead');
      head.append(el('h2', null, `Group ${g.id}`));
      head.append(el('span', 'chip' + (g.kind === 'exact' ? ' exact' : ''), g.kind === 'exact' ? 'exact copies' : 'similar'));
      head.append(el('span', 'why', `suggested keeper: ${g.reason}`));
      const warn = el('span', 'warn'); head.append(warn);
      const save = el('span', 'save'); head.append(save);
      card.append(head);
      const tiles = el('div', 'tiles');
      for (const m of g.members) {
        const t = el('div', 'tile ' + state.get(m.path));
        t.title = m.path;
        t.append(el('span', 'badge', state.get(m.path).toUpperCase()));
        const im = el('div', 'img');
        if (m.thumb) { const i = new Image(); i.src = m.thumb; i.alt = m.label; i.loading = 'lazy'; im.append(i); }
        else im.append(el('span', null, 'no preview'));
        t.append(im);
        const meta = el('div', 'meta');
        const cut = m.label.lastIndexOf('/');
        meta.append(el('div', 'name', m.label.slice(cut + 1)));
        meta.append(el('div', 'dir', cut >= 0 ? m.label.slice(0, cut) : '.'));
        const r1 = el('div', 'row'); r1.append(el('span', null, `${m.w}×${m.h}`), el('span', null, fmtBytes(m.bytes))); meta.append(r1);
        const r2 = el('div', 'row');
        r2.append(el('span', null, `sharp ${m.sharp}`));
        if (m.exact) r2.append(el('span', 'tag', 'identical bytes'));
        else if (m === g.members[0]) r2.append(el('span', null, 'reference'));
        else r2.append(el('span', null, `Δp ${m.pd} · Δd ${m.dd}`));
        meta.append(r2);
        t.append(meta);
        t.addEventListener('click', () => {
          state.set(m.path, state.get(m.path) === 'keep' ? 'move' : 'keep');
          render();
        });
        tiles.append(t);
      }
      card.append(tiles);
      const kept = g.members.filter(m => state.get(m.path) === 'keep').length;
      const moved = g.members.filter(m => state.get(m.path) === 'move');
      warn.textContent = kept === 0 ? 'Keep at least one photo in this group' : '';
      save.textContent = moved.length ? `frees ${fmtBytes(moved.reduce((s, m) => s + m.bytes, 0))}` : 'nothing to move';
      root.append(card);
    }
    updateStats();
  }

  function updateStats() {
    let move = 0, bytes = 0, bad = 0;
    for (const g of DATA.groups) {
      let kept = 0;
      for (const m of g.members) {
        if (state.get(m.path) === 'move') { move++; bytes += m.bytes; } else kept++;
      }
      if (!kept) bad++;
    }
    document.getElementById('s-groups').textContent = DATA.groups.length;
    document.getElementById('s-move').textContent = move;
    document.getElementById('s-bytes').textContent = fmtBytes(bytes);
    const btn = document.getElementById('download');
    btn.disabled = bad > 0;
    btn.title = bad ? `${bad} group(s) have no photo kept` : 'Save your choices for twinshot apply';
  }

  document.getElementById('filter').addEventListener('change', render);
  document.getElementById('reset').addEventListener('click', () => { suggested(); render(); });
  document.getElementById('download').addEventListener('click', () => {
    const plan = JSON.parse(JSON.stringify(DATA.plan));
    plan.edited = new Date().toISOString();
    for (const g of plan.groups) {
      const all = [...g.keep, ...g.move];
      g.keep = all.filter(p => state.get(p) === 'keep');
      g.move = all.filter(p => state.get(p) === 'move');
    }
    const blob = new Blob([JSON.stringify(plan, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob); a.download = 'twinshot-plan.json';
    document.body.append(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  });
  render();
})();
</script>
</body>
</html>
"""
