from __future__ import annotations
from dataclasses import asdict, dataclass
from math import exp
from PIL import Image
from app.services.feature_layer.ela import ELAFeatures, extract_ela_features
from app.services.feature_layer.metadata import MetadataFeatures, extract_metadata_features
from app.services.feature_layer.normalization import normalize_feature_mapping
from app.services.feature_layer.statistics import (
    StatisticalArtifactFeatures,
    extract_statistical_features,
)

FEATURE_BIAS = -1.35
FEATURE_WEIGHTS = {
    "ela_mean_residual": 0.90,
    "ela_std_residual": 0.80,
    "ela_hotspot_ratio": 1.10,
    "metadata_missing_exif": 0.30,
    "metadata_missing_camera_data": 0.15, #lowering this weighting because some legitimate images can be missing camera data
    "metadata_software_marker": 0.80,
    "stats_channel_mean_gap": 0.35,
    "stats_channel_std_gap": 0.45,
    "stats_noise_inconsistency": 0.95,
}


@dataclass(slots=True)
class RawFeatureVector:
    # Raw vectors keep the original measurements so the ML workspace can fit its
    # own normalization profile from training data instead of losing signal.
    ela_mean_residual: float
    ela_std_residual: float
    ela_hotspot_ratio: float
    metadata_missing_exif: float
    metadata_missing_camera_data: float
    metadata_software_marker: float
    stats_channel_mean_gap: float
    stats_channel_std_gap: float
    stats_noise_inconsistency: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(slots=True)
class FeatureVector:
    # Normalized vectors are the training-ready representation consumed by the
    # lightweight feature classifier and the backend heuristic scorer.
    ela_mean_residual: float
    ela_std_residual: float
    ela_hotspot_ratio: float
    metadata_missing_exif: float
    metadata_missing_camera_data: float
    metadata_software_marker: float
    stats_channel_mean_gap: float
    stats_channel_std_gap: float
    stats_noise_inconsistency: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(slots=True)
class FeatureVectorBundle:
    raw: RawFeatureVector
    normalized: FeatureVector


@dataclass(slots=True)
class FeatureAnalysisResult:
    confidence: float
    raw_vector: RawFeatureVector
    vector: FeatureVector
    signals: list[dict[str, str | float]]
    component_scores: dict[str, float]


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + exp(-value))


def _score_feature_vector(vector: FeatureVector) -> float:
    logit = FEATURE_BIAS
    for name, value in vector.as_dict().items():
        logit += FEATURE_WEIGHTS[name] * value
    return round(_sigmoid(logit), 4)


def _build_raw_feature_vector(
    ela: ELAFeatures,
    metadata: MetadataFeatures,
    statistics: StatisticalArtifactFeatures,
) -> RawFeatureVector:
    return RawFeatureVector(
        ela_mean_residual=ela.mean_residual,
        ela_std_residual=ela.std_residual,
        ela_hotspot_ratio=ela.hotspot_ratio,
        metadata_missing_exif=metadata.missing_exif_score,
        metadata_missing_camera_data=metadata.missing_camera_data_score,
        metadata_software_marker=metadata.software_marker_score,
        stats_channel_mean_gap=statistics.channel_mean_gap,
        stats_channel_std_gap=statistics.channel_std_gap,
        stats_noise_inconsistency=statistics.noise_inconsistency,
    )


def _normalize_raw_vector(raw_vector: RawFeatureVector) -> FeatureVector:
    normalized_values = normalize_feature_mapping(raw_vector.as_dict())
    return FeatureVector(**normalized_values)


def _score_ela_component(vector: FeatureVector) -> float:
    return round(
        (0.40 * vector.ela_mean_residual)
        + (0.35 * vector.ela_std_residual)
        + (0.25 * vector.ela_hotspot_ratio),
        4,
    )


def _score_metadata_component(vector: FeatureVector) -> float:
    return round(
        (0.30 * vector.metadata_missing_exif)
        + (0.25 * vector.metadata_missing_camera_data)
        + (0.45 * vector.metadata_software_marker),
        4,
    )


def _score_statistical_component(vector: FeatureVector) -> float:
    return round(
        (0.20 * vector.stats_channel_mean_gap)
        + (0.25 * vector.stats_channel_std_gap)
        + (0.55 * vector.stats_noise_inconsistency),
        4,
    )


def extract_feature_vector_bundle(image: Image.Image) -> FeatureVectorBundle:
    # This is the shared entrypoint for training and inference. It guarantees
    # both environments see the same raw measurements and normalized feature order.
    ela = extract_ela_features(image)
    metadata = extract_metadata_features(image)
    statistics = extract_statistical_features(image)

    raw_vector = _build_raw_feature_vector(ela, metadata, statistics)
    normalized_vector = _normalize_raw_vector(raw_vector)
    return FeatureVectorBundle(raw=raw_vector, normalized=normalized_vector)


def _compression_signal(ela_score: float) -> dict[str, str | float]:
    if ela_score >= 0.65:
        detail = (
            "Compression anomalies detected. The JPEG recompression pass "
            "produced concentrated error hotspots that can happen after edits."
        )
    elif ela_score >= 0.35:
        detail = (
            "Compression behavior is mixed. The image shows moderate "
            "recompression variation that may come from edits or repeated saves."
        )
    else:
        detail = (
            "Compression anomalies are limited. ELA did not find strong "
            "localized recompression differences."
        )

    return {
        "name": "compression_anomalies",
        "detail": detail,
        "score": round(ela_score, 4),
    }


def _camera_metadata_signal(metadata: MetadataFeatures) -> dict[str, str | float]:
    if metadata.has_camera_data:
        detail = (
            "Camera metadata is present. The image retains camera-identifying "
            "EXIF fields such as make or model."
        )
    else:
        detail = (
            "Camera data cannot be detected. Missing EXIF camera fields are "
            "common for screenshots, edited exports, and many generated images."
        )

    return {
        "name": "camera_metadata",
        "detail": detail,
        "score": round(metadata.missing_camera_data_score, 4),
    }


def _statistical_signal(
    statistics_score: float,
) -> dict[str, str | float]:
    if statistics_score >= 0.60:
        detail = (
            "Color and noise irregularities detected. Channel balance and local "
            "noise distribution vary more than expected."
        )
    elif statistics_score >= 0.35:
        detail = (
            "Statistical artifact signals are moderate. The image shows some "
            "color or noise inconsistency, but not at an extreme level."
        )
    else:
        detail = (
            "Statistical artifact signals are limited. Color balance and local "
            "noise look relatively even."
        )

    return {
        "name": "statistical_artifacts",
        "detail": detail,
        "score": round(statistics_score, 4),
    }


def _software_metadata_signal(metadata: MetadataFeatures) -> dict[str, str | float] | None:
    if metadata.software_marker_score <= 0.0:
        return None

    software_value = metadata.software_tag_value or "unknown software"
    return {
        "name": "software_metadata",
        "detail": (
            f"Software metadata is present: '{software_value}'. Export or editor "
            "metadata can be useful provenance context for the final verdict."
        ),
        "score": round(metadata.software_marker_score, 4),
    }


def analyze_feature_layer(image: Image.Image) -> FeatureAnalysisResult:
    # The feature layer stays intentionally lightweight: it extracts interpretable
    # forensic cues now and keeps the numeric vector ready for later classifier training.
    ela = extract_ela_features(image)
    metadata = extract_metadata_features(image)
    statistics = extract_statistical_features(image)

    raw_vector = _build_raw_feature_vector(ela, metadata, statistics)
    vector = _normalize_raw_vector(raw_vector)
    confidence = _score_feature_vector(vector)
    ela_score = _score_ela_component(vector)
    metadata_score = _score_metadata_component(vector)
    statistical_score = _score_statistical_component(vector)

    signals = [
        _compression_signal(ela_score),
        _camera_metadata_signal(metadata),
        _statistical_signal(statistical_score),
    ]
    software_signal = _software_metadata_signal(metadata)
    if software_signal is not None:
        signals.append(software_signal)

    return FeatureAnalysisResult(
        confidence=confidence,
        raw_vector=raw_vector,
        vector=vector,
        signals=signals,
        component_scores={
            "ela": ela_score,
            "metadata": metadata_score,
            "statistical": statistical_score,
        },
    )
