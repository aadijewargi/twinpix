from pathlib import Path

import pytest

from twinpix import scanner
from twinpix.cache import HashCache
from twinpix.scanner import iter_images, scan


def as_sets(groups, root):
    return {frozenset(str(Path(p.path).relative_to(root.resolve())).replace("\\", "/")
                      for p in g.members) for g in groups}


def test_recovers_ground_truth_groups(library):
    root, truth = library
    result = scan(root, workers=1)
    expected = {frozenset(files) for files in truth.values() if len(files) > 1}
    assert as_sets(result.groups, root) == expected
    assert not result.errors


def test_keeper_is_never_a_downscaled_or_blurred_copy(library):
    root, _ = library
    for g in scan(root, workers=1).groups:
        assert "WhatsApp" not in g.keeper.path
        assert "BURST002" not in g.keeper.path  # the motion-blurred frame
        assert g.keeper.pixels == max(p.pixels for p in g.members)


def test_parallel_matches_serial(library):
    root, _ = library
    serial = scan(root, workers=1)
    parallel = scan(root, workers=2)
    assert as_sets(serial.groups, root) == as_sets(parallel.groups, root)


def test_identical_files_are_flagged(library):
    root, truth = library
    result = scan(root, workers=1)
    backups = [p for p in result.photos if "Backup" in p.path]
    assert backups, "fixture should contain byte-identical backups"
    for b in backups:
        assert b.digest is not None


def test_threshold_zero_still_groups_identical_bytes(library):
    root, _ = library
    result = scan(root, threshold=0, workers=1)
    assert all(g.max_distance == 0 for g in result.groups)
    group_of = {Path(p.path).name + "|" + Path(p.path).parent.name: i
                for i, g in enumerate(result.groups) for p in g.members}
    backups = [k for k in group_of if k.endswith("|Backup")]
    assert backups
    for b in backups:
        assert group_of[b] == group_of[b.replace("|Backup", "|Camera")]


def test_corrupt_and_non_image_files(tmp_path):
    (tmp_path / "notes.txt").write_text("hello")
    (tmp_path / "broken.jpg").write_bytes(b"\xff\xd8\xff\xe0 definitely not a jpeg")
    (tmp_path / "empty.png").write_bytes(b"")
    result = scan(tmp_path, workers=1)
    assert len(result.photos) == 2  # the .txt is ignored
    assert len(result.errors) == 2
    assert result.groups == []


def test_empty_folder(tmp_path):
    result = scan(tmp_path, workers=1)
    assert result.photos == [] and result.groups == [] and result.reclaimable_bytes == 0


def test_missing_folder(tmp_path):
    with pytest.raises(NotADirectoryError):
        scan(tmp_path / "nope")


def test_unicode_and_spaces_in_paths(tmp_path):
    from twinpix.demo import make_scene

    img = make_scene(5).resize((400, 300))
    d = tmp_path / "Fotos Ürlaub 2026" / "日本"
    d.mkdir(parents=True)
    img.save(d / "café photo.jpg", quality=90)
    img.resize((200, 150)).save(d / "café photo (small).JPG", quality=70)
    result = scan(tmp_path, workers=1)
    assert len(result.groups) == 1 and len(result.groups[0].members) == 2


def test_hidden_and_excluded_folders_are_skipped(tmp_path):
    from twinpix.demo import make_scene

    img = make_scene(5).resize((200, 150))
    for sub in (".thumbnails", "review", "keep"):
        (tmp_path / sub).mkdir()
        img.save(tmp_path / sub / "x.jpg")
    found = list(iter_images(tmp_path, exclude=[tmp_path / "review"]))
    assert [p.parent.name for p in found] == ["keep"]


def test_cache_skips_rehashing(library, tmp_path, monkeypatch):
    root, _ = library
    cache = HashCache(tmp_path / "c.sqlite")
    first = scan(root, workers=1, cache=cache)
    calls = []
    real = scanner.hash_file
    monkeypatch.setattr(scanner, "hash_file", lambda p: calls.append(p) or real(p))
    second = scan(root, workers=1, cache=cache)
    assert calls == []
    assert as_sets(first.groups, root) == as_sets(second.groups, root)
    cache.close()


def test_bad_algo_rejected(library):
    with pytest.raises(ValueError):
        scanner.group_photos([], algo="md5")


def test_pure_byte_copies_form_an_identical_group(tmp_path):
    import shutil

    from twinpix.demo import make_scene

    make_scene(8).resize((400, 300)).save(tmp_path / "a.jpg", quality=90)
    shutil.copy(tmp_path / "a.jpg", tmp_path / "a - Copy.jpg")
    (g,) = scan(tmp_path, workers=1).groups
    assert g.identical and g.max_distance == 0 and len(g.members) == 2


def test_many_identical_hashes_group_in_linear_time():
    from twinpix.model import Photo
    from twinpix.scanner import group_photos

    photos = [Photo(f"black_{i:05d}.jpg", 1000 + i, 0, 100, 100, 0, 0, 1.0) for i in range(20000)]
    photos.append(Photo("other.jpg", 5, 0, 100, 100, (1 << 64) - 1, (1 << 64) - 1, 1.0))
    (g,) = group_photos(photos, threshold=10)
    assert len(g.members) == 20000
