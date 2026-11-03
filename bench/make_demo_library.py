"""Build a small, realistic messy photo folder for demos and screenshots.

    python bench/make_demo_library.py DEMO_DIR

Uses scikit-image's bundled sample photos (pip install scikit-image) and
recreates the ways duplicates sneak into a real library: burst shots, chat-app
re-saves, "(1)" download copies, edits, screenshots, plus a corrupt file.
"""

from __future__ import annotations

import os
import sys
import time

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth import _crop, _shift, _watermark, procedural_scene, skimage_photos  # noqa: E402

DAY = 86400


def big(img: Image.Image, width: int = 1600) -> Image.Image:
    return img.resize((width, round(width * img.height / img.width)), Image.BICUBIC)


def save(root: str, rel: str, img: Image.Image, age_days: float, **kw) -> None:
    path = os.path.join(root, *rel.split("/"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path, **kw)
    t = time.time() - age_days * DAY
    os.utime(path, (t, t))


def main(root: str) -> None:
    p = {k: big(v) for k, v in skimage_photos().items()}
    if not p:
        sys.exit("scikit-image is required: pip install scikit-image")

    # A burst of the cat: one sharp frame, two slightly shaken, one shifted.
    cat = p["chelsea"]
    save(root, "2024-05 Home/IMG_3101.jpg", cat.filter(ImageFilter.GaussianBlur(2.2)), 180, quality=90)
    save(root, "2024-05 Home/IMG_3102.jpg", _shift(cat, 0.01), 180, quality=90)
    save(root, "2024-05 Home/IMG_3103.jpg", cat, 180, quality=92)
    save(root, "2024-05 Home/IMG_3104.jpg", _shift(cat, 0.02).filter(ImageFilter.GaussianBlur(1.4)), 180, quality=90)

    # Coffee: original, a WhatsApp re-save, and a "(1)" download copy.
    coffee = p["coffee"]
    save(root, "2024-06 Lisbon/IMG_2210.jpg", coffee, 150, quality=93)
    save(root, "WhatsApp Images/IMG-20240611-WA0007.jpg", coffee.resize((800, 533)), 140, quality=55)
    src = os.path.join(root, "2024-06 Lisbon", "IMG_2210.jpg")
    dup = os.path.join(root, "Downloads", "IMG_2210 (1).jpg")
    os.makedirs(os.path.dirname(dup), exist_ok=True)
    with open(src, "rb") as a, open(dup, "wb") as b:
        b.write(a.read())
    t = time.time() - 20 * DAY
    os.utime(dup, (t, t))

    # Rocket: original, a brightened edit, and a cropped + watermarked share.
    rocket = p["rocket"]
    save(root, "2024-06 Lisbon/IMG_2245.jpg", rocket, 148, quality=93)
    save(root, "Edited/IMG_2245_bright.jpg", ImageEnhance.Brightness(rocket).enhance(1.2), 120, quality=88)
    save(root, "Shared/rocket_story.jpg", _watermark(_crop(rocket, 0.04)), 110, quality=70)

    # Motorcycle: original and a screenshot of it (PNG, smaller).
    moto = p["stereo_motorcycle"]
    save(root, "2024-07 Roadtrip/IMG_4410.jpg", moto, 120, quality=92)
    save(root, "Screenshots/Screenshot 2024-07-02 181502.png", moto.resize((960, 648)), 118)

    # Astronaut: original and a black-and-white edit.
    astro = p["astronaut"]
    save(root, "Wallpapers/astronaut.jpg", astro, 300, quality=94)
    save(root, "Edited/astronaut_bw.jpg", ImageOps.grayscale(astro), 90, quality=90)

    # Unique photos that must NOT be grouped.
    for i, name in enumerate(["hubble_deep_field", "clock", "coins", "camera", "immunohistochemistry",
                              "retina", "moon", "brick", "grass", "gravel"]):
        if name in p:
            save(root, f"Misc/IMG_{5000 + i}.jpg", p[name], 60 + i, quality=90)
    for s in range(6):
        save(root, f"Drawings/scene_{s}.png", procedural_scene(77 + s, (1200, 900)), 30)

    # A truncated JPEG, like an interrupted copy.
    with open(os.path.join(root, "Misc", "IMG_5999.jpg"), "wb") as fh:
        fh.write(b"\xff\xd8\xff\xe0" + b"\x00" * 200)
    print("demo library written to", root)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "demo/Pictures")
