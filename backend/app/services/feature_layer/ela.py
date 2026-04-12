from __future__ import annotations
from dataclasses import dataclass
from io import BytesIO
import numpy as np
from PIL import Image

from app.services.feature_layer.normalization import FEATURE_NORMALIZATION_SPECS
@dataclass(slots=True)
class ELAFeatures:
    #ela returns raw forensic measurements
    #Normalization is handled in one shared place so training and inference canuse the same feature scales
    mean_residual: float
    std_residual: float
    hotspot_ratio: float

def extract_ela_features(image: Image.Image, jpeg_quality: int = 90) -> ELAFeatures:
    rgb_image = image.convert("RGB")
    #this works by recompressing the image, measuting how the pixels change, and looking for localized hotspots that can indicate editing, 
    #synthetic content, or repeated saves after manipulation.
    buffer = BytesIO()
    rgb_image.save(buffer, format="JPEG", quality=jpeg_quality)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert("RGB")
    
    original = np.asarray(rgb_image, dtype=np.float32)
    compressed = np.asarray(recompressed, dtype=np.float32)
    residual = np.abs(original - compressed).mean(axis=2)

    mean_residual = float(residual.mean())
    std_residual = float(residual.std())
    hotspot_ratio = float((residual > 12.0).mean())

    return ELAFeatures(
        mean_residual=mean_residual,
        std_residual=std_residual,
        hotspot_ratio=hotspot_ratio,
    )


def quick_ela_score(image: Image.Image) -> float:
    # This quick score is a weighted combination of normalized ELA features that provides a single proxy for image manipulation likelihood
    ela = extract_ela_features(image)
    mean_score = FEATURE_NORMALIZATION_SPECS["ela_mean_residual"].normalize(ela.mean_residual)
    std_score = FEATURE_NORMALIZATION_SPECS["ela_std_residual"].normalize(ela.std_residual)
    hotspot_score = FEATURE_NORMALIZATION_SPECS["ela_hotspot_ratio"].normalize(
        ela.hotspot_ratio
    )
    return float(np.clip((0.40 * mean_score) + (0.35 * std_score) + (0.25 * hotspot_score), 0.0, 1.0))
