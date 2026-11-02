import hashlib
import json
from pathlib import Path

from twinpix.actions import move_duplicates, undo
from twinpix.scanner import scan


def snapshot(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*")) if p.is_file()
    }


def test_move_then_undo_restores_everything(library_copy, tmp_path):
    root, _ = library_copy
    before = snapshot(root)
    review = tmp_path / "review"
    result = scan(root, workers=1)
    plans, manifest = move_duplicates(result.groups, root, review)

    assert len(plans) == result.duplicate_count > 0
    for g in result.groups:  # keepers stay, duplicates leave
        assert Path(g.keeper.path).exists()
        for p in g.duplicates:
            assert not Path(p.path).exists()
    moved = json.loads(manifest.read_text(encoding="utf-8"))["moves"]
    assert len(moved) == len(plans)
    assert all(Path(m["to"]).is_relative_to(review.resolve()) for m in moved)
    assert scan(root, workers=1).groups == []  # nothing left to dedupe

    restored, problems = undo(manifest)
    assert problems == [] and restored == len(plans)
    assert snapshot(root) == before
    assert not manifest.exists()  # renamed to *.undone.json


def test_dry_run_touches_nothing(library_copy, tmp_path):
    root, _ = library_copy
    before = snapshot(root)
    plans, manifest = move_duplicates(scan(root, workers=1).groups, root, tmp_path / "r", dry_run=True)
    assert plans and manifest is None
    assert snapshot(root) == before
    assert not (tmp_path / "r").exists()


def test_review_folder_inside_library_is_excluded(library_copy):
    root, _ = library_copy
    review = root / "_review"
    first = scan(root, workers=1)
    move_duplicates(first.groups, root, review)
    second = scan(root, workers=1, exclude=[review])
    assert second.groups == []


def test_name_collisions_get_suffixes(tmp_path):
    from twinpix.demo import make_scene

    root = tmp_path / "lib"
    img = make_scene(2).resize((400, 300))
    (root / "a").mkdir(parents=True)
    img.save(root / "a" / "big.jpg", quality=95)
    img.resize((200, 150)).save(root / "a" / "x.jpg", quality=80)
    review = tmp_path / "review"
    (review / "a").mkdir(parents=True)
    (review / "a" / "x.jpg").write_text("already here")
    plans, _ = move_duplicates(scan(root, workers=1).groups, root, review)
    assert [p.dst.name for p in plans] == ["x (1).jpg"]
    assert (review / "a" / "x.jpg").read_text() == "already here"


def test_undo_refuses_to_overwrite(library_copy, tmp_path):
    root, _ = library_copy
    plans, manifest = move_duplicates(scan(root, workers=1).groups, root, tmp_path / "r")
    plans[0].src.write_text("a new file took this name")
    restored, problems = undo(manifest)
    assert restored == len(plans) - 1
    assert len(problems) == 1 and "occupied" in problems[0]
    assert plans[0].src.read_text() == "a new file took this name"
    assert manifest.exists()  # kept so the remaining file can still be found
