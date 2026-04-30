from __future__ import annotations
from dataclasses import dataclass
from io import BytesIO
import numpy as np
from PIL import Image

from app.services.feature_layer.normalization import FEATURE_NORMALIZATION_SPECS
#added three JPEC qualities to compare against as many online images have different JPEG qualities.
DEFAULT_JPEG_QUALITIES = (85, 90, 95)
HOTSPOT_STD_FACTOR = 2.0
EPSILON = 1e-8

@dataclass(slots=True)
class ELAFeatures:
    mean_residual: float
    std_residual: float
    hotspot_ratio: float
    p95_residual: float #how high the residual is at the 95th percentile
    p99_residual: float #how extreme the residual is at the 99th percentile
    hotspot_residual_share: float #how much of the total error is concentrated in hotspots.

def _residual_for_quality(rgb_image: Image.Image, jpeg_quality: int) -> np.ndarray:
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be between 1 and 100.")

    #this works by recompressing the image, measuring how the pixels change, and looking for localized hotspots that can indicate editing,
    #synthetic content, or repeated saves after manipulation.
    buffer = BytesIO()
    rgb_image.save(buffer, format="JPEG", quality=jpeg_quality)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert("RGB")

    original = np.asarray(rgb_image, dtype=np.float32)
    compressed = np.asarray(recompressed, dtype=np.float32)
    return np.abs(original - compressed).mean(axis=2)


def _summarize_residual(residual: np.ndarray) -> dict[str, float]:
    mean_residual = float(residual.mean())
    std_residual = float(residual.std())
    threshold = mean_residual + (HOTSPOT_STD_FACTOR * std_residual)
    hotspot_mask = residual > threshold
    residual_sum = float(residual.sum())
    hotspot_residual_share = (
        0.0 if residual_sum <= EPSILON else float(residual[hotspot_mask].sum() / residual_sum)
    )

    return {
        "mean_residual": mean_residual,
        "std_residual": std_residual,
        "hotspot_ratio": float(hotspot_mask.mean()),
        "p95_residual": float(np.percentile(residual, 95)),
        "p99_residual": float(np.percentile(residual, 99)),
        "hotspot_residual_share": hotspot_residual_share,
    }

def _mean_metric(metrics: list[dict[str, float]], metric_name: str) -> float:
    return float(np.mean([metric[metric_name] for metric in metrics]))

def extract_ela_features(image: Image.Image, jpeg_quality: int | None = None) -> ELAFeatures:
    rgb_image = image.convert("RGB")
    qualities = (jpeg_quality,) if jpeg_quality is not None else DEFAULT_JPEG_QUALITIES
    residual_metrics = [
        _summarize_residual(_residual_for_quality(rgb_image, quality))
        for quality in qualities
    ]

    return ELAFeatures(
        mean_residual=_mean_metric(residual_metrics, "mean_residual"),
        std_residual=_mean_metric(residual_metrics, "std_residual"),
        hotspot_ratio=_mean_metric(residual_metrics, "hotspot_ratio"),
        p95_residual=_mean_metric(residual_metrics, "p95_residual"),
        p99_residual=_mean_metric(residual_metrics, "p99_residual"),
        hotspot_residual_share=_mean_metric(residual_metrics, "hotspot_residual_share"),
    )

def quick_ela_score(image: Image.Image) -> float:
    # This quick score is a weighted combination of normalized ELA features that provides a single proxy for image manipulation likelihood
    ela = extract_ela_features(image)
    mean_score = FEATURE_NORMALIZATION_SPECS["ela_mean_residual"].normalize(ela.mean_residual)
    std_score = FEATURE_NORMALIZATION_SPECS["ela_std_residual"].normalize(ela.std_residual)
    hotspot_score = FEATURE_NORMALIZATION_SPECS["ela_hotspot_ratio"].normalize(
        ela.hotspot_ratio
    )
    p95_score = FEATURE_NORMALIZATION_SPECS["ela_p95_residual"].normalize(ela.p95_residual)
    p99_score = FEATURE_NORMALIZATION_SPECS["ela_p99_residual"].normalize(ela.p99_residual)
    hotspot_share_score = FEATURE_NORMALIZATION_SPECS[
        "ela_hotspot_residual_share"
    ].normalize(ela.hotspot_residual_share)
    return float(
        np.clip(
            (0.20 * mean_score)
            + (0.18 * std_score)
            + (0.20 * hotspot_score)
            + (0.16 * p95_score)
            + (0.16 * p99_score)
            + (0.10 * hotspot_share_score),
            0.0,
            1.0,
        )
    )
