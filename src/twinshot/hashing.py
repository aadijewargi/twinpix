"""Perceptual hashes and image fingerprints.

Three 64-bit hashes are implemented from scratch on top of Pillow + NumPy:

* ``ahash`` (average hash): 8x8 grayscale thumbnail, bit = pixel > mean.
* ``dhash`` (difference hash): 9x8 thumbnail, bit = pixel brighter than its
  right-hand neighbour. Encodes horizontal gradients; very robust to
  brightness/contrast changes and re-compression.
* ``phash`` (DCT hash): 32x32 thumbnail -> 2-D DCT-II -> keep the 8x8 block of
  lowest frequencies, bit = coefficient > median. Robust to resizing, JPEG
  artefacts and mild blur because those mostly touch high frequencies.

Two images are "near duplicates" when the Hamming distance between their hashes
(the number of differing bits) is small.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Optional, Union

import numpy as np
from PIL import Image, ImageOps

PathLike = Union[str, "os.PathLike[str]"]

# Big panoramas are legitimate photos; we only ever read a downscaled draft.
Image.MAX_IMAGE_PIXELS = None

# Bump when hashing changes so stale cache entries are ignored.
ALGO_VERSION = 1

_RESAMPLE = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS


def popcount(x: int) -> int:
    """Number of set bits (works on Python 3.9, where int.bit_count is missing)."""
    return bin(x).count("1")


def hamming(a: int, b: int) -> int:
    """Hamming distance between two integer hashes."""
    return popcount(a ^ b)


def _bits_to_int(bits: np.ndarray) -> int:
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bool(bit))
    return value


def to_gray_array(img: Image.Image, size: tuple) -> np.ndarray:
    """Resize to ``size`` (w, h) and return float64 luminance in [0, 255]."""
    gray = img.convert("L").resize(size, _RESAMPLE)
    return np.asarray(gray, dtype=np.float64)


def ahash(img: Image.Image, hash_size: int = 8) -> int:
    px = to_gray_array(img, (hash_size, hash_size))
    return _bits_to_int(px > px.mean())


def dhash(img: Image.Image, hash_size: int = 8) -> int:
    px = to_gray_array(img, (hash_size + 1, hash_size))
    return _bits_to_int(px[:, 1:] > px[:, :-1])


@lru_cache(maxsize=4)
def _dct_matrix(n: int) -> np.ndarray:
    """Orthonormal DCT-II matrix C such that DCT(x) = C @ x."""
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    c = np.cos(np.pi * (2 * i + 1) * k / (2 * n)) * np.sqrt(2.0 / n)
    c[0, :] = np.sqrt(1.0 / n)
    return c


def dct2(block: np.ndarray) -> np.ndarray:
    """2-D orthonormal DCT-II of a square block."""
    c = _dct_matrix(block.shape[0])
    return c @ block @ c.T


def phash(img: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> int:
    n = hash_size * highfreq_factor
    px = to_gray_array(img, (n, n))
    low = dct2(px)[:hash_size, :hash_size]
    return _bits_to_int(low > np.median(low))


def sharpness(img: Image.Image, side: int = 256) -> float:
    """Contrast-normalised variance of the Laplacian.

    Higher means more fine detail; used to pick the best frame from a burst.
    Computed at a fixed resolution so a downscaled copy and the original are
    compared on equal footing, and divided by the image's own intensity
    variance so a brightened or contrast-boosted edit does not look "sharper"
    than the original just because its gradients got bigger.
    """
    px = to_gray_array(img, (side, side))
    lap = (
        -4.0 * px[1:-1, 1:-1]
        + px[:-2, 1:-1]
        + px[2:, 1:-1]
        + px[1:-1, :-2]
        + px[1:-1, 2:]
    )
    return float(100.0 * lap.var() / (px.var() + 1e-6))


def colorfulness(img: Image.Image, side: int = 64) -> float:
    """Hasler & Suesstrunk (2003) colourfulness; ~0 for black-and-white images."""
    rgb = np.asarray(img.convert("RGB").resize((side, side), _RESAMPLE), dtype=np.float64)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    rg, yb = r - g, 0.5 * (r + g) - b
    return float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))


def _flatten_alpha(img: Image.Image) -> Image.Image:
    """Convert any mode to RGB, compositing transparency onto white."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.getchannel("A"))
        return bg
    return img.convert("RGB")


def open_image(path: PathLike, draft_size: Optional[tuple] = (512, 512)):
    """Open an image, returning ``(rgb_image, (width, height))``.

    ``width, height`` are the *display* dimensions of the original file (EXIF
    rotation applied). For JPEGs, Pillow's draft mode lets libjpeg decode at
    1/2, 1/4 or 1/8 scale, which makes hashing large photos several times
    faster: we never need more than a few hundred pixels.
    """
    with Image.open(path) as im:
        width, height = im.size
        try:
            orientation = im.getexif().get(0x0112, 1)
        except Exception:  # malformed EXIF should not sink the whole file
            orientation = 1
        if orientation in (5, 6, 7, 8):
            width, height = height, width
        if draft_size is not None and im.format == "JPEG":
            im.draft("RGB", draft_size)
        im.load()
        try:
            im = ImageOps.exif_transpose(im)
        except Exception:
            pass
        rgb = _flatten_alpha(im)
    return rgb, (width, height)


@dataclass
class Fingerprint:
    """Everything twinshot needs to know about one photo."""

    path: str
    bytes: int
    mtime: float
    width: int
    height: int
    phash: int
    dhash: int
    sharpness: float
    colorfulness: float = 0.0

    @property
    def pixels(self) -> int:
        return self.width * self.height

    def to_dict(self) -> dict:
        d = asdict(self)
        # 64-bit ints as hex strings: JSON/JavaScript numbers lose precision.
        d["phash"] = f"{self.phash:016x}"
        d["dhash"] = f"{self.dhash:016x}"
        return d


def fingerprint(path: PathLike) -> Fingerprint:
    """Hash one file. Raises on unreadable/corrupt images."""
    path = os.fspath(path)
    st = os.stat(path)
    rgb, (w, h) = open_image(path)
    return Fingerprint(
        path=path,
        bytes=st.st_size,
        mtime=st.st_mtime,
        width=w,
        height=h,
        phash=phash(rgb),
        dhash=dhash(rgb),
        sharpness=sharpness(rgb),
        colorfulness=colorfulness(rgb),
    )
