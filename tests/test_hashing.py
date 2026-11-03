import numpy as np
import pytest
from PIL import Image, ImageEnhance, ImageFilter

from twinshot.hashing import (
    _dct_matrix, ahash, colorfulness, dct2, dhash, fingerprint, hamming, open_image, phash,
    popcount, sharpness,
)

from conftest import jpeg_roundtrip, scene

HASHES = [ahash, dhash, phash]


def test_popcount_and_hamming_match_reference():
    rng = np.random.default_rng(0)
    for _ in range(200):
        a, b = (int(x) for x in rng.integers(0, 2**63, 2, dtype=np.int64))
        assert popcount(a) == bin(a).count("1")
        assert hamming(a, b) == bin(a ^ b).count("1")
    assert hamming(0, 2**64 - 1) == 64


def test_dct_matrix_is_orthonormal():
    c = _dct_matrix(32)
    assert np.allclose(c @ c.T, np.eye(32), atol=1e-12)


def test_dct2_matches_definition():
    rng = np.random.default_rng(1)
    x = rng.random((8, 8))
    n = 8
    ref = np.zeros((n, n))
    for u in range(n):
        for v in range(n):
            au = np.sqrt((1 if u == 0 else 2) / n)
            av = np.sqrt((1 if v == 0 else 2) / n)
            s = 0.0
            for i in range(n):
                for j in range(n):
                    s += x[i, j] * np.cos(np.pi * (2 * i + 1) * u / (2 * n)) * np.cos(np.pi * (2 * j + 1) * v / (2 * n))
            ref[u, v] = au * av * s
    assert np.allclose(dct2(x), ref)


@pytest.mark.parametrize("h", HASHES)
def test_hashes_are_64_bit_and_deterministic(h):
    img = scene(5)
    v = h(img)
    assert 0 <= v < 2**64
    assert h(img.copy()) == v


@pytest.mark.parametrize("h", [dhash, phash])
@pytest.mark.parametrize(
    "edit",
    [
        lambda im: im.resize((im.width // 2, im.height // 2)),
        lambda im: im.resize((im.width // 4, im.height // 4)),
        lambda im: jpeg_roundtrip(im, 20),
        lambda im: ImageEnhance.Brightness(im).enhance(1.2),
        lambda im: im.filter(ImageFilter.GaussianBlur(1.5)),
        lambda im: im.convert("L").convert("RGB"),
    ],
    ids=["half", "quarter", "jpeg20", "brighter", "blur", "gray"],
)
def test_edited_copies_stay_close(h, edit):
    for seed in range(5):
        img = scene(seed)
        assert hamming(h(img), h(edit(img))) <= 10


@pytest.mark.parametrize("h", [dhash, phash])
def test_different_photos_are_far_apart(h):
    hs = [h(scene(s)) for s in range(12)]
    dists = [hamming(hs[i], hs[j]) for i in range(12) for j in range(i + 1, 12)]
    assert min(dists) > 14
    assert 24 <= np.mean(dists) <= 40  # ~32 for independent hashes


def test_mirror_is_not_a_match():
    # Documented limitation: perceptual hashes are not flip-invariant.
    img = scene(7)
    assert hamming(phash(img), phash(img.transpose(Image.FLIP_LEFT_RIGHT))) > 12


def test_sharpness_ranks_blur_and_ignores_brightness():
    for seed in range(6):
        img = scene(seed)
        s = sharpness(img)
        assert s > sharpness(img.filter(ImageFilter.GaussianBlur(1.5))) * 1.5
        # brightness/contrast edits must stay inside the 25% "same sharpness" band
        assert sharpness(ImageEnhance.Brightness(img).enhance(1.25)) < s * 1.25
        assert sharpness(ImageEnhance.Contrast(img).enhance(1.25)) < s * 1.25


def test_colorfulness_separates_gray_from_colour():
    img = scene(9)
    assert colorfulness(img.convert("L").convert("RGB")) < 1e-6
    assert colorfulness(img) > 10


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L", "LA", "P", "1", "CMYK", "I;16"])
def test_open_image_handles_all_modes(tmp_path, mode):
    img = scene(3).convert("RGBA" if mode in ("RGBA", "LA") else "RGB")
    if mode == "I;16":
        conv = Image.fromarray(np.asarray(img.convert("L"), dtype=np.uint16) * 256)
    else:
        conv = img.convert(mode)
    ext = "tiff" if mode == "CMYK" else "png"
    path = tmp_path / f"x.{ext}"
    conv.save(path)
    rgb, (w, h) = open_image(path)
    assert rgb.mode == "RGB" and (w, h) == conv.size
    fp = fingerprint(path)
    assert fp.width == w and fp.height == h


def test_tiny_image(tmp_path):
    p = tmp_path / "dot.png"
    Image.new("RGB", (1, 1), (200, 10, 10)).save(p)
    fp = fingerprint(p)
    assert (fp.width, fp.height) == (1, 1)


def test_exif_rotation_is_applied(tmp_path):
    img = scene(4, size=(400, 300))
    exif = Image.Exif()
    exif[0x0112] = 6  # "rotate 90 CW to display"
    p = tmp_path / "rot.jpg"
    img.save(p, quality=95, exif=exif.tobytes())
    fp = fingerprint(p)
    assert (fp.width, fp.height) == (300, 400)
    upright = img.transpose(Image.ROTATE_270)
    assert hamming(fp.phash, phash(upright)) <= 4


def test_corrupt_file_raises(tmp_path):
    p = tmp_path / "bad.jpg"
    p.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 50)
    with pytest.raises(Exception):
        fingerprint(p)


def test_fingerprint_to_dict_uses_hex():
    from twinshot.hashing import Fingerprint

    d = Fingerprint("x", 1, 0.0, 2, 3, 2**63 + 5, 7, 1.0).to_dict()
    assert d["phash"] == "8000000000000005" and d["dhash"] == "0000000000000007"
