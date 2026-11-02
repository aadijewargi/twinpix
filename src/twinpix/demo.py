"""Generate a synthetic photo library full of realistic duplicate traps.

    python -m twinpix.demo demo_photos --scenes 40

Each procedurally drawn "scene" (landscape, skyline, still life, or abstract)
is saved once as an original, and some get the kinds of copies that pile up in
a real camera roll:

* ``resized``  - a half-size copy re-saved by a messenger app
* ``lowq``     - the same pixels recompressed at JPEG quality 35
* ``burst``    - three burst frames: slightly shifted, one with motion blur
* ``edited``   - brightness/contrast tweak
* ``backup``   - a byte-identical copy in a Backup folder
* ``png``      - a lossless PNG export
* ``crop``     - a 4 % crop
* ``rotated``  - pixels stored sideways with an EXIF orientation tag

Ground truth is written to ``truth.json`` so accuracy can be measured.
Everything is generated, so the demo has no licensing strings attached.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

W, H = 1600, 1200


def _lerp(c1, c2, t):
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


def _gradient(rng, top, bottom) -> Image.Image:
    t = np.linspace(0, 1, H)[:, None, None]
    arr = np.array(top)[None, None, :] * (1 - t) + np.array(bottom)[None, None, :] * t
    arr = np.repeat(arr, W, axis=1)
    return Image.fromarray(arr.astype(np.uint8))


def _rand_color(rng, lo=0, hi=255):
    return tuple(int(x) for x in rng.integers(lo, hi, 3))


def _landscape(rng) -> Image.Image:
    img = _gradient(rng, _rand_color(rng, 60, 255), _rand_color(rng, 120, 255))
    d = ImageDraw.Draw(img)
    sx, sy, sr = rng.integers(100, W - 100), rng.integers(80, H // 2), rng.integers(40, 140)
    d.ellipse([sx - sr, sy - sr, sx + sr, sy + sr], fill=_rand_color(rng, 180, 255))
    for layer in range(int(rng.integers(2, 5))):
        base = int(H * (0.35 + 0.15 * layer + rng.uniform(-0.05, 0.05)))
        xs = np.linspace(0, W, int(rng.integers(5, 14)))
        ys = base + rng.normal(0, 90, len(xs))
        pts = [(0, H)] + [(float(x), float(y)) for x, y in zip(xs, ys)] + [(W, H)]
        d.polygon(pts, fill=_rand_color(rng, 10, 160))
    for _ in range(int(rng.integers(0, 25))):
        x, y = int(rng.integers(0, W)), int(rng.integers(H // 2, H))
        s = int(rng.integers(20, 70))
        d.polygon([(x, y - 3 * s), (x - s, y), (x + s, y)], fill=_rand_color(rng, 0, 90))
    return img


def _skyline(rng) -> Image.Image:
    img = _gradient(rng, _rand_color(rng, 0, 120), _rand_color(rng, 100, 255))
    d = ImageDraw.Draw(img)
    x = int(rng.integers(-50, 0))
    while x < W:
        bw, bh = int(rng.integers(60, 220)), int(rng.integers(200, 900))
        col = _rand_color(rng, 10, 110)
        d.rectangle([x, H - bh, x + bw, H], fill=col)
        win = _rand_color(rng, 150, 255)
        for wy in range(H - bh + 20, H - 20, 40):
            for wx in range(x + 10, x + bw - 20, 30):
                if rng.random() < 0.45:
                    d.rectangle([wx, wy, wx + 12, wy + 18], fill=win)
        x += bw + int(rng.integers(0, 40))
    return img


def _still_life(rng) -> Image.Image:
    img = Image.new("RGB", (W, H), _rand_color(rng, 120, 240))
    d = ImageDraw.Draw(img)
    table = int(H * rng.uniform(0.5, 0.75))
    d.rectangle([0, table, W, H], fill=_rand_color(rng, 40, 160))
    for _ in range(int(rng.integers(3, 8))):
        cx, r = int(rng.integers(150, W - 150)), int(rng.integers(60, 220))
        cy = table - int(r * rng.uniform(0.3, 1.0))
        col = _rand_color(rng)
        if rng.random() < 0.5:
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)
        else:
            d.rectangle([cx - r // 2, cy - 2 * r, cx + r // 2, table], fill=col)
    return img


def _abstract(rng) -> Image.Image:
    img = Image.new("RGB", (W, H), _rand_color(rng))
    d = ImageDraw.Draw(img)
    for _ in range(int(rng.integers(15, 40))):
        x0, y0 = int(rng.integers(-200, W)), int(rng.integers(-200, H))
        x1, y1 = x0 + int(rng.integers(80, 700)), y0 + int(rng.integers(80, 700))
        shape = rng.integers(0, 3)
        col = _rand_color(rng)
        if shape == 0:
            d.ellipse([x0, y0, x1, y1], fill=col)
        elif shape == 1:
            d.rectangle([x0, y0, x1, y1], fill=col)
        else:
            d.line([x0, y0, x1, y1], fill=col, width=int(rng.integers(10, 60)))
    return img


SCENES = [_landscape, _skyline, _still_life, _abstract]


def make_scene(seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    img = SCENES[seed % len(SCENES)](rng)
    img = img.filter(ImageFilter.GaussianBlur(1.2))
    # Sensor noise and vignetting make the pictures behave like photos.
    arr = np.asarray(img, dtype=np.float64)
    yy, xx = np.mgrid[0:H, 0:W]
    r2 = ((xx - W / 2) / W) ** 2 + ((yy - H / 2) / H) ** 2
    arr *= (1 - 0.6 * r2)[..., None]
    arr += rng.normal(0, 6, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _shift(img: Image.Image, dx: int, dy: int) -> Image.Image:
    """The camera moved a little: crop a slightly offset window, scale back up."""
    m = 20  # margin in pixels, larger than any shift we use
    box = (m + dx, m + dy, img.width - m + dx, img.height - m + dy)
    return img.crop(box).resize(img.size, Image.Resampling.BICUBIC)


def build(out: Path, scenes: int = 40, seed: int = 7) -> dict:
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    rng = np.random.default_rng(seed)
    folders = ["Camera", "WhatsApp Images", "Downloads", "Backup", "Edits"]
    for f in folders:
        (out / f).mkdir(parents=True)
    truth: dict[str, list[str]] = {}

    def save(img: Image.Image, rel: str, **kw) -> str:
        path = out / rel
        img.save(path, **kw)
        return rel

    for i in range(scenes):
        img = make_scene(seed * 1000 + i)
        name = f"IMG_{2040 + i:04d}"
        group = [save(img, f"Camera/{name}.jpg", quality=92)]
        if i % 3 == 2:  # one third of scenes are unique: no copies at all
            truth[name] = group
            continue
        kinds = rng.choice(
            ["resized", "lowq", "burst", "edited", "backup", "png", "crop", "rotated"],
            size=int(rng.integers(1, 4)), replace=False,
        )
        for kind in kinds:
            if kind == "resized":
                small = img.resize((W // 2, H // 2), Image.Resampling.LANCZOS)
                group.append(save(small, f"WhatsApp Images/IMG-2026-WA{i:04d}.jpg", quality=80))
            elif kind == "lowq":
                group.append(save(img, f"Downloads/{name}_compressed.jpg", quality=35))
            elif kind == "burst":
                for b in range(1, 4):
                    frame = _shift(img, int(rng.integers(-14, 15)), int(rng.integers(-10, 11)))
                    if b == 2:
                        frame = frame.filter(ImageFilter.BoxBlur(5))
                    group.append(save(frame, f"Camera/{name}_BURST{b:03d}.jpg", quality=92))
            elif kind == "edited":
                e = ImageEnhance.Brightness(img).enhance(float(rng.uniform(1.08, 1.25)))
                e = ImageEnhance.Contrast(e).enhance(float(rng.uniform(1.05, 1.2)))
                group.append(save(e, f"Edits/{name}_edit.jpg", quality=90))
            elif kind == "backup":
                shutil.copy2(out / group[0], out / f"Backup/{name}.jpg")
                group.append(f"Backup/{name}.jpg")
            elif kind == "png":
                group.append(save(img, f"Edits/{name}.png"))
            elif kind == "crop":
                c = img.crop((int(W * 0.02), int(H * 0.02), int(W * 0.98), int(H * 0.98)))
                group.append(save(c, f"Edits/{name}_crop.jpg", quality=90))
            elif kind == "rotated":
                rot = img.transpose(Image.Transpose.ROTATE_90)
                exif = Image.Exif()
                exif[0x0112] = 6  # "rotate 90 CW to display" undoes the CCW ROTATE_90 above
                group.append(save(rot, f"Downloads/{name}_phone.jpg", quality=90, exif=exif))
        truth[name] = group
    (out / "truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    return truth


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--scenes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    truth = build(args.out, args.scenes, args.seed)
    n = sum(len(v) for v in truth.values())
    print(f"wrote {n} photos ({len(truth)} scenes) to {args.out}")


if __name__ == "__main__":
    main()
