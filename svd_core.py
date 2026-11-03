"""Image preparation and truncated-SVD helpers for the Pixel Fox app."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_SOURCE_PIXELS = 40_000_000
MAX_ANALYSIS_SIDE = 512
SUPPORTED_FORMATS = {"PNG", "JPEG", "WEBP", "GIF", "BMP", "TIFF"}


class ImageInputError(ValueError):
    """An image upload that cannot be safely or correctly processed."""


@dataclass(frozen=True)
class ImageSVD:
    original: np.ndarray
    singular_values: tuple[np.ndarray, ...]
    left_vectors: tuple[np.ndarray, ...]
    right_vectors_t: tuple[np.ndarray, ...]
    width: int
    height: int
    source_width: int
    source_height: int
    source_format: str
    grayscale: bool
    was_resized: bool

    @property
    def max_rank(self) -> int:
        return int(self.singular_values[0].shape[0])


def _open_supported_image(image_bytes: bytes) -> tuple[Image.Image, str, int, int]:
    if not image_bytes:
        raise ImageInputError("That file is empty. Choose a PNG, JPEG, WebP, GIF, BMP, or TIFF image.")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise ImageInputError("This file is larger than 20 MB. Choose a smaller image and try again.")

    try:
        with Image.open(BytesIO(image_bytes)) as probe:
            source_format = (probe.format or "").upper()
            source_width, source_height = probe.size
            if source_format not in SUPPORTED_FORMATS:
                raise ImageInputError(
                    "This image format is not supported. Choose PNG, JPEG, WebP, GIF, BMP, or TIFF."
                )
            if source_width < 1 or source_height < 1:
                raise ImageInputError("The image has invalid dimensions. Choose another image.")
            if source_width * source_height > MAX_SOURCE_PIXELS:
                raise ImageInputError(
                    "This image has more than 40 million pixels. Resize it before uploading so it can be processed safely."
                )
            probe.verify()

        with Image.open(BytesIO(image_bytes)) as image:
            image.seek(0)  # Animated GIF/TIFF inputs use their first frame.
            image = ImageOps.exif_transpose(image)
            return image.copy(), source_format, image.width, image.height
    except ImageInputError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, EOFError, SyntaxError) as exc:
        raise ImageInputError(
            "We couldn't read that image. It may be damaged or incomplete; try exporting it as PNG or JPEG."
        ) from exc
    except Image.DecompressionBombError as exc:
        raise ImageInputError(
            "This image is too large to open safely. Resize it before uploading and try again."
        ) from exc


def _to_analysis_image(image: Image.Image) -> tuple[Image.Image, bool]:
    # Transparent pixels are composited onto white so the SVD works on visible RGB values.
    has_alpha = image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info)
    if has_alpha:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        background.alpha_composite(rgba)
        image = background.convert("RGB")

    grayscale = image.mode in {"1", "L", "I", "F", "I;16"}
    image = image.convert("L" if grayscale else "RGB")
    source_size = image.size
    image.thumbnail((MAX_ANALYSIS_SIDE, MAX_ANALYSIS_SIDE), Image.Resampling.LANCZOS)
    return image, image.size != source_size


def decompose_image(image_bytes: bytes) -> ImageSVD:
    """Validate, prepare and factor an uploaded image channel by channel."""
    image, source_format, source_width, source_height = _open_supported_image(image_bytes)
    image, was_resized = _to_analysis_image(image)
    pixels = np.asarray(image, dtype=np.uint8)
    height, width = pixels.shape[:2]
    grayscale = pixels.ndim == 2
    channels = (pixels,) if grayscale else tuple(pixels[:, :, channel] for channel in range(3))

    factors = []
    for channel in channels:
        u, singular_values, vh = np.linalg.svd(channel.astype(np.float32), full_matrices=False)
        factors.append((u, singular_values, vh))

    return ImageSVD(
        original=pixels,
        singular_values=tuple(item[1] for item in factors),
        left_vectors=tuple(item[0] for item in factors),
        right_vectors_t=tuple(item[2] for item in factors),
        width=width,
        height=height,
        source_width=source_width,
        source_height=source_height,
        source_format=source_format,
        grayscale=grayscale,
        was_resized=was_resized,
    )


def reconstruct_image(decomposition: ImageSVD, k: int) -> np.ndarray:
    """Return the clipped uint8 reconstruction using the first ``k`` terms."""
    if not 1 <= int(k) <= decomposition.max_rank:
        raise ValueError(f"k must be between 1 and {decomposition.max_rank}.")
    channels = []
    for u, singular_values, vh in zip(
        decomposition.left_vectors,
        decomposition.singular_values,
        decomposition.right_vectors_t,
    ):
        rank = int(k)
        reconstruction = (u[:, :rank] * singular_values[:rank]) @ vh[:rank, :]
        channels.append(np.clip(np.rint(reconstruction), 0, 255).astype(np.uint8))
    if decomposition.grayscale:
        return channels[0]
    return np.stack(channels, axis=2)


def retained_energy_percent(decomposition: ImageSVD, k: int) -> float:
    """Fraction of squared singular-value energy retained across all channels."""
    total = sum(float(np.dot(values, values)) for values in decomposition.singular_values)
    retained = sum(float(np.dot(values[:k], values[:k])) for values in decomposition.singular_values)
    return 100.0 if total == 0 else 100.0 * retained / total


def estimated_factor_bytes(decomposition: ImageSVD, k: int) -> int:
    """Approximate float32 storage for the retained U, S, and V factors."""
    channel_count = len(decomposition.singular_values)
    per_channel = k * (decomposition.height + decomposition.width + 1)
    return int(per_channel * channel_count * np.dtype(np.float32).itemsize)


def encode_reconstruction(pixels: np.ndarray, output_format: str) -> bytes:
    """Encode an RGB or grayscale reconstruction in a supported download format."""
    normalized_format = output_format.upper()
    if normalized_format not in {"PNG", "JPEG", "WEBP"}:
        raise ValueError("Choose PNG, JPEG, or WebP for the download.")
    image = Image.fromarray(pixels)
    if normalized_format == "JPEG" and image.mode not in {"L", "RGB"}:
        image = image.convert("RGB")
    output = BytesIO()
    options = {"quality": 90, "method": 4} if normalized_format == "WEBP" else {}
    if normalized_format == "JPEG":
        options = {"quality": 90, "optimize": True}
    image.save(output, format=normalized_format, **options)
    return output.getvalue()
