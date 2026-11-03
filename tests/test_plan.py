import json
import os

import pytest

from twinshot.grouping import ScanResult, find_groups
from twinshot.hashing import fingerprint
from twinshot.plan import (
    PlanError, _relative_destination, apply_plan, build_plan, load_plan, save_plan, undo, validate_plan,
)
from twinshot.scanner import iter_image_files


def make_plan(root):
    fps = [fingerprint(p) for p in iter_image_files([str(root)])]
    res = ScanResult(groups=find_groups(fps), scanned=len(fps))
    return build_plan(res, [str(root)])


def all_files(root):
    return sorted(os.path.relpath(os.path.join(d, f), root) for d, _, fs in os.walk(root) for f in fs)


def test_apply_then_undo_restores_everything(library, tmp_path):
    root, truth = library
    before = all_files(root)
    plan = make_plan(root)
    q = tmp_path / "quarantine"
    res = apply_plan(plan, quarantine_dir=str(q))
    assert len(res.moved) == sum(len(v) - 1 for v in truth.values())
    for paths in truth.values():
        assert os.path.exists(paths[0])  # keepers untouched
        for p in paths[1:]:
            assert not os.path.exists(p)
    # quarantine mirrors the folder layout
    assert (q / "chat" / "a_small.jpg").exists()
    restored = undo(str(q))
    assert len(restored.moved) == len(res.moved) and not restored.skipped
    assert all_files(root) == before


def test_dry_run_moves_nothing(library, tmp_path):
    root, _ = library
    before = all_files(root)
    res = apply_plan(make_plan(root), quarantine_dir=str(tmp_path / "q"), dry_run=True)
    assert res.moved and res.manifest is None
    assert all_files(root) == before and not (tmp_path / "q").exists()


def test_plan_roundtrip_and_validation(library, tmp_path):
    root, _ = library
    plan = make_plan(root)
    path = tmp_path / "plan.json"
    save_plan(plan, str(path))
    assert load_plan(str(path)) == plan

    bad = json.loads(json.dumps(plan))
    g = bad["groups"][0]
    g["move"], g["keep"] = g["keep"] + g["move"], []
    with pytest.raises(PlanError, match="keep at least one"):
        validate_plan(bad)

    bad = json.loads(json.dumps(plan))
    bad["groups"][0]["move"].append(bad["groups"][0]["keep"][0])
    with pytest.raises(PlanError, match="both kept and moved"):
        validate_plan(bad)

    with pytest.raises(PlanError):
        validate_plan({"groups": []})


def test_changed_file_is_skipped(library, tmp_path):
    root, truth = library
    plan = make_plan(root)
    victim = truth["a"][1]
    with open(victim, "ab") as fh:
        fh.write(b"edited after scan")
    res = apply_plan(plan, quarantine_dir=str(tmp_path / "q"))
    assert (victim, "file changed since the scan") in res.skipped
    assert os.path.exists(victim)


def test_group_skipped_when_keeper_vanished(library, tmp_path):
    root, truth = library
    plan = make_plan(root)
    os.remove(truth["b"][0])
    res = apply_plan(plan, quarantine_dir=str(tmp_path / "q"))
    assert os.path.exists(truth["b"][1])
    assert any(p == truth["b"][1] for p, _ in res.skipped)


def test_missing_file_is_skipped(library, tmp_path):
    root, truth = library
    plan = make_plan(root)
    os.remove(truth["c"][1])
    res = apply_plan(plan, quarantine_dir=str(tmp_path / "q"))
    assert (truth["c"][1], "file no longer exists") in res.skipped


def test_name_collisions_in_quarantine(tmp_path):
    from twinshot.plan import _unique

    p = tmp_path / "x.jpg"
    p.write_bytes(b"1")
    (tmp_path / "x (1).jpg").write_bytes(b"2")
    assert _unique(str(p)) == str(tmp_path / "x (2).jpg")
    assert _unique(str(tmp_path / "new.jpg")) == str(tmp_path / "new.jpg")


def test_undo_does_not_overwrite(library, tmp_path):
    root, truth = library
    q = tmp_path / "q"
    apply_plan(make_plan(root), quarantine_dir=str(q))
    clash = truth["c"][1]
    with open(clash, "wb") as fh:
        fh.write(b"new file at old path")
    res = undo(str(q / "manifest.jsonl"))
    assert any(p == clash for p, _ in res.skipped)
    with open(clash, "rb") as fh:
        assert fh.read() == b"new file at old path"
    # the manifest keeps the entry that could not be restored
    with open(q / "manifest.jsonl", encoding="utf-8") as fh:
        assert len([l for l in fh if l.strip()]) == 1


def test_relative_destination(tmp_path):
    r1, r2 = str(tmp_path / "Photos"), str(tmp_path / "Phone")
    p = os.path.join(r1, "2024", "a.jpg")
    assert _relative_destination(p, [r1]) == os.path.join("2024", "a.jpg")
    assert _relative_destination(p, [r1, r2]) == os.path.join("Photos", "2024", "a.jpg")
    outside = _relative_destination(str(tmp_path / "elsewhere" / "b.jpg"), [r1])
    assert outside.startswith("_outside") and outside.endswith("b.jpg")
