from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NormalizationSpec:
    #feature extraction produces raw numeric measurements. These specs convert
    #them into bounded 0..1 values that the lightweight classifier can consume
    #consistently across the backend and ML workspace.
    lower: float
    upper: float

    def normalize(self, value: float) -> float:
        if self.upper <= self.lower:
            raise ValueError("Normalization upper bound must be greater than lower bound.")
        normalized = (value - self.lower) / (self.upper - self.lower)
        return max(0.0, min(1.0, float(normalized)))

FEATURE_NORMALIZATION_SPECS: dict[str, NormalizationSpec] = {
    "ela_mean_residual": NormalizationSpec(lower=0.0, upper=18.0),
    "ela_std_residual": NormalizationSpec(lower=0.0, upper=18.0),
    "ela_hotspot_ratio": NormalizationSpec(lower=0.0, upper=0.25),
    "ela_p95_residual": NormalizationSpec(lower=0.0, upper=35.0),
    "ela_p99_residual": NormalizationSpec(lower=0.0, upper=55.0),
    "ela_hotspot_residual_share": NormalizationSpec(lower=0.0, upper=0.60),
    "metadata_missing_exif": NormalizationSpec(lower=0.0, upper=1.0),
    "metadata_missing_camera_data": NormalizationSpec(lower=0.0, upper=1.0),
    "metadata_software_marker": NormalizationSpec(lower=0.0, upper=1.0),
    "stats_channel_mean_gap": NormalizationSpec(lower=0.0, upper=0.35),
    "stats_channel_std_gap": NormalizationSpec(lower=0.0, upper=0.20),
    "stats_noise_inconsistency": NormalizationSpec(lower=0.0, upper=1.50),
}

FEATURE_NAMES: tuple[str, ...] = tuple(FEATURE_NORMALIZATION_SPECS.keys())

def normalize_feature_mapping(raw_values: dict[str, float]) -> dict[str, float]:
    missing_features = set(FEATURE_NAMES) - set(raw_values)
    if missing_features:
        raise KeyError(
            "Missing raw feature values for normalization: "
            + ", ".join(sorted(missing_features))
        )

    return {
        feature_name: FEATURE_NORMALIZATION_SPECS[feature_name].normalize(
            raw_values[feature_name]
        )
        for feature_name in FEATURE_NAMES
    }
