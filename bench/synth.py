"""Benchmark image generator: distinct base photos and realistic variants.

Base images come from two sources so no download is needed:

* scikit-image's bundled sample photos (if installed): each photo whole plus
  its four quadrants, upscaled, giving several distinct "shots" per photo;
* procedural scenes: sky gradient, smooth noise "terrain" and random shapes,
  deterministic per seed.

Variants mimic how duplicates really appear in a photo library.
"""

from __future__ import annotations

import io
import random
from typing import Callable, Dict, List

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps


def procedural_scene(seed: int, size=(640, 480)) -> Image.Image:
    rng = np.random.default_rng(seed)
    w, h = size
    top, bottom = rng.integers(0, 256, 3), rng.integers(0, 256, 3)
    t = np.linspace(0, 1, h)[:, None, None]
    img = (top * (1 - t) + bottom * t) * np.ones((1, w, 1))
    # smooth value noise: upsample a coarse random grid
    coarse = rng.random((6, 8, 3)) * 160 - 80
    noise = np.asarray(
        Image.fromarray(((coarse + 80) / 160 * 255).astype(np.uint8)).resize(size, Image.BICUBIC),
        dtype=np.float64,
    ) / 255 * 160 - 80
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    r = random.Random(seed)
    for _ in range(r.randint(4, 10)):
        x0, y0 = r.randint(-50, w), r.randint(-50, h)
        x1, y1 = x0 + r.randint(30, w // 2), y0 + r.randint(30, h // 2)
        color = tuple(r.randint(0, 255) for _ in range(3))
        kind = r.random()
        if kind < 0.4:
            draw.ellipse([x0, y0, x1, y1], fill=color)
        elif kind < 0.7:
            draw.rectangle([x0, y0, x1, y1], fill=color)
        else:
            pts = [(r.randint(0, w), r.randint(0, h)) for _ in range(3)]
            draw.polygon(pts, fill=color)
    return pil.filter(ImageFilter.GaussianBlur(1.2))


SKIMAGE_PHOTOS = [
    "astronaut", "coffee", "chelsea", "rocket", "camera", "coins", "hubble_deep_field",
    "immunohistochemistry", "retina", "moon", "stereo_motorcycle", "clock", "brick", "grass", "gravel",
]


def skimage_photos() -> Dict[str, Image.Image]:
    try:
        from skimage import data
    except Exception:
        return {}
    out = {}
    for name in SKIMAGE_PHOTOS:
        try:
            arr = getattr(data, name)()
        except Exception:
            continue
        if isinstance(arr, tuple):
            arr = arr[0]
        out[name] = Image.fromarray(arr).convert("RGB")
    return out


def base_images(n_procedural: int = 150) -> Dict[str, Image.Image]:
    bases: Dict[str, Image.Image] = {}
    for name, img in skimage_photos().items():
        w, h = img.size
        bases[name] = img.resize((640, int(640 * h / w)))
        for qi, (x, y) in enumerate([(0, 0), (w // 2, 0), (0, h // 2), (w // 2, h // 2)]):
            q = img.crop((x, y, x + w // 2, y + h // 2))
            bases[f"{name}_q{qi}"] = q.resize((640, int(640 * q.size[1] / q.size[0])), Image.BICUBIC)
    for s in range(n_procedural):
        bases[f"scene{s:03d}"] = procedural_scene(1000 + s)
    return bases


def _jpeg(img: Image.Image, q: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _crop(img: Image.Image, frac: float) -> Image.Image:
    w, h = img.size
    dx, dy = int(w * frac / 2), int(h * frac / 2)
    return img.crop((dx, dy, w - dx, h - dy))


def _shift(img: Image.Image, frac: float) -> Image.Image:
    w, h = img.size
    dx, dy = int(w * frac), int(h * frac / 2)
    return img.crop((dx, dy, w, h)).resize((w, h))


def _watermark(img: Image.Image) -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)
    w, h = out.size
    d.rectangle([w - 190, h - 40, w - 10, h - 12], fill=(255, 255, 255))
    d.text((w - 180, h - 34), "(c) shared via chat", fill=(0, 0, 0))
    return out


Transform = Callable[[Image.Image], Image.Image]

# Variants a user would call "the same photo".
POSITIVE: Dict[str, Transform] = {
    "resize 50%": lambda im: im.resize((im.width // 2, im.height // 2), Image.LANCZOS),
    "thumbnail 25%": lambda im: im.resize((im.width // 4, im.height // 4), Image.LANCZOS),
    "JPEG q=40": lambda im: _jpeg(im, 40),
    "JPEG q=15": lambda im: _jpeg(im, 15),
    "brightness +25%": lambda im: ImageEnhance.Brightness(im).enhance(1.25),
    "contrast -25%": lambda im: ImageEnhance.Contrast(im).enhance(0.75),
    "grayscale": lambda im: ImageOps.grayscale(im).convert("RGB"),
    "blur r=1.5": lambda im: im.filter(ImageFilter.GaussianBlur(1.5)),
    "watermark": _watermark,
    "burst (2% shift)": lambda im: _shift(im, 0.02),
    "crop 5%": lambda im: _crop(im, 0.05),
    "crop 10%": lambda im: _crop(im, 0.10),
    "rotate 3deg": lambda im: im.rotate(3, resample=Image.BICUBIC, expand=False),
}

# Variants perceptual hashes are known NOT to handle (reported honestly).
HARD: Dict[str, Transform] = {
    "crop 25%": lambda im: _crop(im, 0.25),
    "mirror": ImageOps.mirror,
    "rotate 90deg (pixels)": lambda im: im.rotate(90, expand=True),
}


def variants(img: Image.Image, include_hard: bool = True) -> List[tuple]:
    out = [(name, fn(img)) for name, fn in POSITIVE.items()]
    if include_hard:
        out += [(name, fn(img)) for name, fn in HARD.items()]
    return out
