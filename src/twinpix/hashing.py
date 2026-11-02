"""Perceptual image hashes.

Two classic 64-bit hashes are implemented with nothing but Pillow and NumPy:

* **dHash** (difference hash): shrink to 9x8 grey pixels and record whether
  each pixel is brighter than its right-hand neighbour. Very fast and robust
  to resizing, recompression and brightness/contrast changes.
* **pHash** (DCT hash): shrink to 32x32, take the 2-D DCT, keep the 8x8
  lowest-frequency block and threshold each coefficient against the median.
  Slightly slower, but more robust to small crops, blur and gamma changes.

Two images are "similar" when the Hamming distance between their hashes is
small (out of 64 bits). Typical thresholds: 0-4 near-identical, 5-10 same
shot re-saved or resized, >20 unrelated.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, ImageOps

HASH_BITS = 64
_RESAMPLE = Image.Resampling.LANCZOS


def hamming(a: int, b: int) -> int:
    """Number of differing bits between two hashes."""
    return (a ^ b).bit_count()


def _bits_to_int(bits: np.ndarray) -> int:
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bool(bit))
    return value


def flatten(img: Image.Image) -> Image.Image:
    """Convert any Pillow mode to 8-bit "L" or "RGB", flattening alpha onto white.

    Flattening matters: a transparent PNG and the same picture exported as a
    JPEG (which has no alpha) should hash the same. 16-bit and float images
    are stretched to the 0-255 range.
    """
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    if img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        arr = np.asarray(img, dtype=np.float64)
        lo, hi = float(arr.min()), float(arr.max())
        scale = 255.0 / (hi - lo) if hi > lo else 0.0
        return Image.fromarray(((arr - lo) * scale).astype(np.uint8))
    if img.mode in ("L", "RGB"):
        return img
    return img.convert("L" if img.mode == "1" else "RGB")


def to_grey(img: Image.Image) -> Image.Image:
    """Any Pillow image as 8-bit grey (alpha flattened onto white)."""
    return flatten(img).convert("L")


def dhash(img: Image.Image, hash_size: int = 8) -> int:
    """Difference hash of a Pillow image (``hash_size**2`` bits)."""
    grey = to_grey(img).resize((hash_size + 1, hash_size), _RESAMPLE)
    px = np.asarray(grey, dtype=np.int16)
    return _bits_to_int(px[:, 1:] > px[:, :-1])


@lru_cache(maxsize=4)
def _dct_matrix(n: int) -> np.ndarray:
    """Orthonormal DCT-II matrix, so that ``D @ x`` is the DCT of ``x``."""
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    mat = np.cos(np.pi * (2 * i + 1) * k / (2 * n)) * np.sqrt(2.0 / n)
    mat[0, :] = np.sqrt(1.0 / n)
    return mat


def phash(img: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> int:
    """DCT-based perceptual hash of a Pillow image (``hash_size**2`` bits)."""
    n = hash_size * highfreq_factor
    grey = to_grey(img).resize((n, n), _RESAMPLE)
    px = np.asarray(grey, dtype=np.float64)
    d = _dct_matrix(n)
    coeffs = d @ px @ d.T
    low = coeffs[:hash_size, :hash_size]
    # The DC term (mean brightness) would dominate the median, so leave it out.
    median = np.median(low.ravel()[1:])
    return _bits_to_int(low > median)


COLOUR_GRID = 4


def colour_signature(img: Image.Image) -> bytes:
    """Chromaticity of a 4x4 grid of cells, 32 bytes.

    Perceptual hashes only see brightness, so two different pictures with a
    similar light/dark layout (a ball on a table, say) can collide. Each cell
    stores its (r, g) chromaticity, r/(r+g+b) and g/(r+g+b), which ignores
    overall brightness, so an exposure-edited copy still matches while a
    differently coloured scene does not.
    """
    rgb = flatten(img).convert("RGB")
    small = rgb.resize((COLOUR_GRID, COLOUR_GRID), Image.Resampling.BOX)
    arr = np.asarray(small, dtype=np.float64) + 1.0
    chroma = arr[..., :2] / arr.sum(axis=-1, keepdims=True)
    return bytes(np.round(chroma * 255).astype(np.uint8).ravel())


def colour_distance(a: bytes, b: bytes) -> float:
    """Mean absolute chromaticity difference between two signatures (0-1)."""
    x = np.frombuffer(a, dtype=np.uint8).astype(np.float64)
    y = np.frombuffer(b, dtype=np.uint8).astype(np.float64)
    return float(np.abs(x - y).mean() / 255.0)


def sharpness(img: Image.Image, max_side: int = 512) -> float:
    """Contrast-normalised variance of the Laplacian on a downscaled grey copy.

    Variance of the Laplacian is a standard focus measure: blurry or shaky
    frames in a burst score low. It also grows with the square of image
    contrast, so it is divided by the pixel variance; otherwise a copy with
    boosted contrast would look "sharper" than the original. Downscaling
    first keeps it fast and makes it ignore JPEG block noise. The result is
    scaled by 1000 to give friendlier numbers.
    """
    grey = to_grey(img)
    grey.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)
    px = np.asarray(grey, dtype=np.float64)
    if px.shape[0] < 3 or px.shape[1] < 3:
        return 0.0
    lap = (
        px[:-2, 1:-1] + px[2:, 1:-1] + px[1:-1, :-2] + px[1:-1, 2:] - 4.0 * px[1:-1, 1:-1]
    )
    spread = px.var()
    if spread < 1e-6:  # a flat image has no edges at all
        return 0.0
    return float(1000.0 * lap.var() / spread)


def open_image(path: Union[str, Path], target: int = 512) -> tuple[Image.Image, tuple[int, int]]:
    """Open an image fast, honouring EXIF rotation.

    Returns the (possibly reduced) image plus the *original* pixel size.
    For JPEGs, ``Image.draft`` asks libjpeg to decode at 1/2, 1/4 or 1/8
    scale, which makes hashing a 24-megapixel photo several times faster.
    """
    img = Image.open(path)
    width, height = img.size
    # EXIF orientations 5-8 store the picture sideways; report displayed size.
    if img.getexif().get(0x0112, 1) in (5, 6, 7, 8):
        width, height = height, width
    img.draft("RGB", (target, target))
    img = ImageOps.exif_transpose(img)
    img.load()
    return img, (width, height)
