import json

from twinpix.cli import main


def test_scan_writes_report_and_json(library, tmp_path, capsys):
    root, truth = library
    report, out = tmp_path / "r.html", tmp_path / "r.json"
    assert main(["scan", str(root), "-j", "1", "--report", str(report), "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert "duplicates" in text and "keep" in text
    html = report.read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>") and "data:image/jpeg;base64," in html
    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data["groups"]) == sum(1 for v in truth.values() if len(v) > 1)


def test_move_and_undo_commands(library_copy, tmp_path, capsys):
    root, _ = library_copy
    review = tmp_path / "review"
    assert main(["move", str(root), "--to", str(review), "-j", "1", "--dry-run"]) == 0
    assert not review.exists()
    assert main(["move", str(root), "--to", str(review), "-j", "1"]) == 0
    manifest = next(review.glob("twinpix-manifest-*.json"))
    assert main(["undo", str(manifest)]) == 0
    assert "restored" in capsys.readouterr().out


def test_errors(tmp_path, capsys):
    assert main(["scan", str(tmp_path / "missing"), "--no-report"]) == 2
    try:
        main(["scan", str(tmp_path), "--threshold", "99"])
    except SystemExit as exc:
        assert "threshold" in str(exc)
    else:
        raise AssertionError("expected SystemExit")


def test_empty_folder_report(tmp_path):
    report = tmp_path / "r.html"
    assert main(["scan", str(tmp_path), "-q", "--report", str(report)]) == 0
    assert "No near-duplicates found" in report.read_text(encoding="utf-8")
