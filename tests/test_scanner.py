import os

import pytest

import twinshot.scanner as scanner
from twinshot.scanner import FingerprintCache, fingerprint_all, iter_image_files

from conftest import scene


def test_iter_skips_hidden_quarantine_and_non_images(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / ".hidden").mkdir()
    (tmp_path / "twinshot-quarantine").mkdir()
    for d in ("a", ".hidden", "twinshot-quarantine"):
        scene(1, (32, 32)).save(tmp_path / d / "x.jpg")
    (tmp_path / "a" / "notes.txt").write_text("hi")
    scene(2, (32, 32)).save(tmp_path / "UPPER.JPG")
    scene(3, (32, 32)).save(tmp_path / "a" / "ünïcødé 写真.png")
    found = [os.path.relpath(p, tmp_path) for p in iter_image_files([str(tmp_path)])]
    assert sorted(found) == sorted(["UPPER.JPG", os.path.join("a", "x.jpg"), os.path.join("a", "ünïcødé 写真.png")])


def test_iter_non_recursive_and_single_file(tmp_path):
    (tmp_path / "sub").mkdir()
    scene(1, (32, 32)).save(tmp_path / "top.png")
    scene(1, (32, 32)).save(tmp_path / "sub" / "deep.png")
    assert len(list(iter_image_files([str(tmp_path)], recursive=False))) == 1
    single = str(tmp_path / "sub" / "deep.png")
    # same file given twice (folder + file) is only yielded once
    assert len(list(iter_image_files([single, str(tmp_path)]))) == 2


def test_iter_missing_root_is_empty(tmp_path):
    assert list(iter_image_files([str(tmp_path / "nope")])) == []


def test_corrupt_files_are_reported_not_fatal(tmp_path):
    scene(1).save(tmp_path / "ok.jpg")
    (tmp_path / "bad.jpg").write_bytes(b"not an image")
    fps, skipped = fingerprint_all(list(iter_image_files([str(tmp_path)])), jobs=1)
    assert [os.path.basename(f.path) for f in fps] == ["ok.jpg"]
    assert len(skipped) == 1 and skipped[0][0].endswith("bad.jpg")


def test_cache_hits_and_invalidation(tmp_path, monkeypatch):
    img = tmp_path / "p.jpg"
    scene(1).save(img)
    cache = FingerprintCache(str(tmp_path / "cache" / "fp.sqlite"))
    first, _ = fingerprint_all([str(img)], jobs=1, cache=cache)

    def boom(path):
        raise AssertionError("should have been served from cache")

    monkeypatch.setattr(scanner, "fingerprint", boom)
    second, skipped = fingerprint_all([str(img)], jobs=1, cache=cache)
    assert second == first and not skipped

    monkeypatch.undo()
    scene(2).save(img)  # file changes: size/mtime differ -> recomputed
    os.utime(img, (1, 1))
    third, _ = fingerprint_all([str(img)], jobs=1, cache=cache)
    assert third[0].phash != first[0].phash
    cache.close()


def test_parallel_matches_serial(tmp_path):
    for s in range(20):
        scene(s, (64, 48)).save(tmp_path / f"{s:02d}.png")
    paths = list(iter_image_files([str(tmp_path)]))
    serial, _ = fingerprint_all(paths, jobs=1)
    parallel, _ = fingerprint_all(paths, jobs=2)
    assert serial == parallel


def test_progress_callback(tmp_path):
    for s in range(3):
        scene(s, (32, 32)).save(tmp_path / f"{s}.png")
    calls = []
    fingerprint_all(list(iter_image_files([str(tmp_path)])), jobs=1, progress=lambda d, t: calls.append((d, t)))
    assert calls[-1] == (3, 3)


@pytest.mark.parametrize("platform,var", [("win32", "LOCALAPPDATA"), ("linux", "XDG_CACHE_HOME")])
def test_default_cache_path(monkeypatch, tmp_path, platform, var):
    monkeypatch.setattr(scanner.sys, "platform", platform)
    monkeypatch.setenv(var, str(tmp_path))
    assert scanner.default_cache_path().startswith(str(tmp_path))
