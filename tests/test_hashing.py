import io

import numpy as np
import pytest
from PIL import Image, ImageEnhance, ImageFilter

from twinpix.demo import make_scene
from twinpix.hashing import _dct_matrix, dhash, hamming, open_image, phash, sharpness, to_grey


@pytest.fixture(scope="module")
def scene():
    return make_scene(11)


def jpeg_roundtrip(img, quality):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf)


def test_hamming_basics():
    assert hamming(0, 0) == 0
    assert hamming(0, (1 << 64) - 1) == 64
    assert hamming(0b1011, 0b0001) == 2


def test_dct_matrix_is_orthonormal():
    d = _dct_matrix(32)
    assert np.allclose(d @ d.T, np.eye(32), atol=1e-12)


@pytest.mark.parametrize("fn", [dhash, phash])
def test_hash_fits_in_64_bits(fn, scene):
    h = fn(scene)
    assert 0 <= h < 2**64


@pytest.mark.parametrize("fn", [dhash, phash])
def test_hash_is_deterministic(fn, scene):
    assert fn(scene) == fn(scene.copy())


@pytest.mark.parametrize("fn", [dhash, phash])
@pytest.mark.parametrize(
    "transform",
    [
        lambda im: im.resize((im.width // 2, im.height // 2)),
        lambda im: im.resize((im.width // 5, im.height // 5)),
        lambda im: jpeg_roundtrip(im, 30),
        lambda im: ImageEnhance.Brightness(im).enhance(1.2),
        lambda im: ImageEnhance.Contrast(im).enhance(1.15),
        lambda im: im.convert("L"),
        lambda im: im.filter(ImageFilter.GaussianBlur(2)),
    ],
    ids=["half-size", "fifth-size", "jpeg-q30", "brighter", "contrast", "greyscale", "blur"],
)
def test_hash_survives_everyday_edits(fn, transform, scene):
    assert hamming(fn(scene), fn(transform(scene))) <= 8


@pytest.mark.parametrize("fn,margin", [(dhash, 8), (phash, 12)])
def test_unrelated_photos_are_far_apart(fn, margin):
    # dHash only looks at horizontal gradients of a 9x8 thumbnail, so similar
    # compositions can land closer together; that is why pHash is the default.
    hashes = [fn(make_scene(s)) for s in range(100, 112)]
    dists = [hamming(a, b) for i, a in enumerate(hashes) for b in hashes[i + 1 :]]
    assert min(dists) > margin
    assert 20 <= float(np.mean(dists)) <= 44  # roughly half of the 64 bits differ


def test_alpha_is_flattened_onto_white(scene):
    rgba = scene.convert("RGBA")
    arr = np.asarray(rgba).copy()
    arr[:100, :100, 3] = 0  # fully transparent corner
    rgba = Image.fromarray(arr)
    flat = Image.new("RGB", rgba.size, "white")
    flat.paste(rgba, mask=rgba.split()[3])
    assert phash(rgba) == phash(flat)


@pytest.mark.parametrize("mode", ["1", "L", "LA", "P", "RGB", "RGBA", "CMYK", "I;16", "I", "F"])
def test_every_pillow_mode_hashes(mode, scene):
    small = scene.resize((64, 48))
    img = small.convert("L").convert(mode) if mode in ("I;16", "I", "F") else small.convert(mode)
    assert to_grey(img).mode == "L"
    assert 0 <= phash(img) < 2**64 and 0 <= dhash(img) < 2**64


def test_tiny_and_flat_images():
    one = Image.new("RGB", (1, 1), (10, 20, 30))
    flat = Image.new("RGB", (300, 200), (128, 128, 128))
    for img in (one, flat):
        phash(img), dhash(img)
    assert sharpness(one) == 0.0
    assert sharpness(flat) == 0.0


def test_sharpness_drops_with_blur_but_not_with_contrast(scene):
    s0 = sharpness(scene)
    assert sharpness(scene.filter(ImageFilter.BoxBlur(5))) < 0.5 * s0
    assert sharpness(ImageEnhance.Contrast(scene).enhance(1.2)) == pytest.approx(s0, rel=0.15)


def test_open_image_honours_exif_rotation(tmp_path, scene):
    upright = tmp_path / "upright.jpg"
    sideways = tmp_path / "sideways.jpg"
    scene.save(upright, quality=92)
    exif = Image.Exif()
    exif[0x0112] = 6  # display = rotate stored pixels 90 degrees clockwise
    scene.transpose(Image.Transpose.ROTATE_90).save(sideways, quality=92, exif=exif)
    a, size_a = open_image(upright)
    b, size_b = open_image(sideways)
    assert size_a == size_b == scene.size
    assert hamming(phash(a), phash(b)) <= 4


def test_open_image_uses_reduced_jpeg_decode(tmp_path, scene):
    path = tmp_path / "big.jpg"
    scene.save(path, quality=90)
    img, full = open_image(path, target=256)
    assert full == scene.size
    assert img.width < scene.width  # decoded at reduced scale


def test_colour_signature_ignores_exposure_but_not_hue(scene):
    from twinpix.hashing import colour_distance, colour_signature

    base = colour_signature(scene)
    assert len(base) == 32
    brighter = colour_signature(ImageEnhance.Brightness(scene).enhance(1.25))
    smaller = colour_signature(jpeg_roundtrip(scene.resize((320, 240)), 40))
    assert colour_distance(base, brighter) < 0.02
    assert colour_distance(base, smaller) < 0.02
    r, g, b = scene.split()
    swapped = colour_signature(Image.merge("RGB", (b, r, g)))  # same layout, new colours
    assert colour_distance(base, swapped) > 0.04
    assert colour_distance(base, base) == 0.0


@pytest.mark.parametrize("mode", ["1", "L", "LA", "P", "RGBA", "CMYK", "I;16", "F"])
def test_colour_signature_handles_every_mode(mode, scene):
    from twinpix.hashing import colour_signature

    small = scene.resize((64, 48))
    img = small.convert("L").convert(mode) if mode in ("I;16", "F") else small.convert(mode)
    assert len(colour_signature(img)) == 32
