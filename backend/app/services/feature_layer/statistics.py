from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(slots=True)
class StatisticalArtifactFeatures:
    # These are raw statistical measurements. They are normalized later using a
    # shared spec so the same feature definitions can be reused in ML training.
    channel_mean_gap: float
    channel_std_gap: float
    noise_inconsistency: float


def _box_blur(gray_image: np.ndarray) -> np.ndarray:
    padded = np.pad(gray_image, 1, mode="edge")
    return (
        padded[:-2, :-2]
        + padded[:-2, 1:-1]
        + padded[:-2, 2:]
        + padded[1:-1, :-2]
        + padded[1:-1, 1:-1]
        + padded[1:-1, 2:]
        + padded[2:, :-2]
        + padded[2:, 1:-1]
        + padded[2:, 2:]
    ) / 9.0


def _patch_mean_grid(values: np.ndarray) -> np.ndarray:
    height, width = values.shape
    patch_size = max(4, min(height, width) // 8)
    usable_height = max(patch_size, height - (height % patch_size))
    usable_width = max(patch_size, width - (width % patch_size))
    cropped = values[:usable_height, :usable_width]
    return cropped.reshape(
        usable_height // patch_size,
        patch_size,
        usable_width // patch_size,
        patch_size,
    ).mean(axis=(1, 3))


def extract_statistical_features(image: Image.Image) -> StatisticalArtifactFeatures:
    rgb_image = image.convert("RGB")
    rgb = np.asarray(rgb_image, dtype=np.float32) / 255.0

    # Channel distribution gaps are a cheap proxy for color inconsistency. They
    # should not decide the verdict alone, but they add useful forensic texture.
    channel_means = rgb.mean(axis=(0, 1))
    channel_stds = rgb.std(axis=(0, 1))
    channel_mean_gap = float(channel_means.max() - channel_means.min())
    channel_std_gap = float(channel_stds.max() - channel_stds.min())

    # Noise consistency is estimated from the high-frequency residual after a
    # tiny blur. Manipulated regions often have a different residual profile
    # than the rest of the image.
    gray = rgb.mean(axis=2)
    residual = np.abs(gray - _box_blur(gray))
    patch_grid = _patch_mean_grid(residual)
    noise_inconsistency = float(patch_grid.std() / (patch_grid.mean() + 1e-6))

    return StatisticalArtifactFeatures(
        channel_mean_gap=channel_mean_gap,
        channel_std_gap=channel_std_gap,
        noise_inconsistency=noise_inconsistency,
    )
