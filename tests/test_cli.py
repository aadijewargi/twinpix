import json
import os
import re

from twinshot.cli import human_bytes, main


def scan(root, tmp_path, *extra):
    report, plan = tmp_path / "r.html", tmp_path / "p.json"
    rc = main(["scan", str(root), "--report", str(report), "--plan", str(plan), "--no-cache", "-j", "1", *extra])
    return rc, report, plan


def embedded_data(report):
    html = report.read_text(encoding="utf-8")
    m = re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S)
    return json.loads(m.group(1).replace("<\\/", "</"))


def test_scan_apply_undo_cycle(library, tmp_path, capsys):
    root, truth = library
    rc, report, plan = scan(root, tmp_path)
    assert rc == 0
    out = capsys.readouterr().out
    assert "3 duplicate group(s)" in out
    data = embedded_data(report)
    assert len(data["groups"]) == 3
    assert all(m["thumb"].startswith("data:image/jpeg;base64,") for g in data["groups"] for m in g["members"])

    q = tmp_path / "quarantine"
    assert main(["apply", str(plan), "-y", "-q", str(q)]) == 0
    assert "Moved 4 photo(s)" in capsys.readouterr().out
    assert main(["undo", str(q)]) == 0
    assert all(os.path.exists(p) for v in truth.values() for p in v)


def test_apply_dry_run(library, tmp_path, capsys):
    root, truth = library
    _, _, plan = scan(root, tmp_path)
    capsys.readouterr()
    assert main(["apply", str(plan), "--dry-run"]) == 0
    assert "Would move 4 photo(s)" in capsys.readouterr().out
    assert all(os.path.exists(p) for v in truth.values() for p in v)


def test_apply_asks_for_confirmation(library, tmp_path, monkeypatch):
    root, truth = library
    _, _, plan = scan(root, tmp_path)
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    assert main(["apply", str(plan)]) == 1
    assert all(os.path.exists(p) for v in truth.values() for p in v)


def test_empty_folder(tmp_path, capsys):
    (tmp_path / "empty").mkdir()
    rc, report, plan = scan(tmp_path / "empty", tmp_path)
    assert rc == 0 and "0 duplicate group(s)" in capsys.readouterr().out
    assert embedded_data(report)["groups"] == []
    assert main(["apply", str(plan), "-y"]) == 0


def test_missing_path(tmp_path, capsys):
    assert main(["scan", str(tmp_path / "nope")]) == 2


def test_invalid_plan(tmp_path):
    p = tmp_path / "x.json"
    p.write_text("{}")
    assert main(["apply", str(p), "-y"]) == 2


def test_hash_command(library, capsys):
    root, truth = library
    assert main(["hash", truth["a"][0]]) == 0
    assert re.match(r"[0-9a-f]{16}  [0-9a-f]{16}  ", capsys.readouterr().out)


def test_search_engines_agree(library, tmp_path):
    root, _ = library
    results = []
    for engine in ("mih", "brute", "bktree"):
        _, _, plan = scan(root, tmp_path, "--search", engine)
        results.append([sorted(g["keep"] + g["move"]) for g in json.loads(plan.read_text())["groups"]])
    assert results[0] == results[1] == results[2]


def test_report_escapes_script_breakout(tmp_path):
    from twinshot.grouping import ScanResult, find_groups
    from twinshot.hashing import Fingerprint
    from twinshot.plan import build_plan
    from twinshot.report import render_report

    evil = "</script><script>alert(1)</script>"
    fps = [Fingerprint(f"{evil}{i}.jpg", 10, 0.0, 8, 8, 0, 0, 1.0, 1.0) for i in range(2)]
    res = ScanResult(groups=find_groups(fps, check_exact=False), scanned=2)
    html = render_report(res, build_plan(res, ["."]), ["."])
    body = html.split('<script id="data" type="application/json">', 1)[1].split("</script>", 1)[0]
    assert "alert(1)" in body  # the payload is still inside the JSON block...
    assert "</script>" not in body and "<\\/script>" in body  # ...but cannot close it


def test_human_bytes():
    assert human_bytes(512) == "512 B"
    assert human_bytes(2048) == "2.0 KB"
    assert human_bytes(5 * 1024**3) == "5.0 GB"
