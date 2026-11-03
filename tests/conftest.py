import io
import os
import random

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


def scene(seed: int, size=(480, 360)) -> Image.Image:
    """Deterministic photo-like test image: gradient + smooth noise + shapes."""
    rng = np.random.default_rng(seed)
    w, h = size
    top, bottom = rng.integers(0, 256, 3), rng.integers(0, 256, 3)
    t = np.linspace(0, 1, h)[:, None, None]
    base = (top * (1 - t) + bottom * t) * np.ones((1, w, 1))
    coarse = (rng.random((5, 7, 3)) * 255).astype(np.uint8)
    noise = np.asarray(Image.fromarray(coarse).resize(size, Image.BICUBIC), dtype=np.float64) - 128
    img = Image.fromarray(np.clip(base + noise, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    r = random.Random(seed)
    for _ in range(6):
        x0, y0 = r.randint(-w // 8, w), r.randint(-h // 8, h)
        box = [x0, y0, x0 + r.randint(4, max(5, w // 2)), y0 + r.randint(4, max(5, h // 2))]
        color = tuple(r.randint(0, 255) for _ in range(3))
        (d.ellipse if r.random() < 0.5 else d.rectangle)(box, fill=color)
    return img.filter(ImageFilter.GaussianBlur(1.0))


def jpeg_roundtrip(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def set_age(path, days: float) -> None:
    t = 1_700_000_000 - days * 86400
    os.utime(path, (t, t))


@pytest.fixture
def library(tmp_path):
    """A folder with 3 known duplicate groups and 4 unique photos.

    Returns (root, truth) where truth maps group name -> set of paths and
    the first path in each group's list is the photo that should be kept.
    """
    root = tmp_path / "Pictures"
    (root / "trip").mkdir(parents=True)
    (root / "chat").mkdir()
    (root / "edits").mkdir()
    truth = {}

    a = scene(1)
    a.save(root / "trip" / "a_original.jpg", quality=95)
    a.resize((240, 180)).save(root / "chat" / "a_small.jpg", quality=60)
    ImageEnhance.Brightness(a).enhance(1.2).save(root / "edits" / "a_bright.jpg", quality=90)
    truth["a"] = [str(root / "trip" / "a_original.jpg"), str(root / "chat" / "a_small.jpg"),
                  str(root / "edits" / "a_bright.jpg")]

    b = scene(2)
    b.save(root / "trip" / "b_sharp.png")
    b.filter(ImageFilter.GaussianBlur(2.5)).save(root / "trip" / "b_blurry.png")
    truth["b"] = [str(root / "trip" / "b_sharp.png"), str(root / "trip" / "b_blurry.png")]

    c = scene(3)
    c.save(root / "trip" / "c.jpg", quality=92)
    with open(root / "trip" / "c.jpg", "rb") as src, open(root / "chat" / "c (1).jpg", "wb") as dst:
        dst.write(src.read())
    truth["c"] = [str(root / "trip" / "c.jpg"), str(root / "chat" / "c (1).jpg")]

    for s in (10, 11, 12, 13):
        scene(s).save(root / "trip" / f"unique_{s}.jpg", quality=90)

    # Make originals the oldest files.
    for i, (name, paths) in enumerate(truth.items()):
        for k, p in enumerate(paths):
            set_age(p, 100 - k)
    return root, truth
