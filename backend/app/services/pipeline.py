from dataclasses import dataclass
from typing import Any

from PIL import Image

from app.services.aggregation import combine_scores
from app.services.feature_layer import analyze_feature_layer
from app.services.model_layer import estimate_cnn_confidence


@dataclass(slots=True)
class DetectionPipelineResult:
    final_confidence: float
    feature_confidence: float
    cnn_confidence: float | None
    signals: list[dict[str, str | float]]
    meta: dict[str, Any]


def run_detection_pipeline(image: Image.Image) -> DetectionPipelineResult:
    # Centralizing orchestration here keeps the route layer small and makes it
    # easier to swap in real model inference as the ML work matures.
    feature_result = analyze_feature_layer(image)
    cnn_score = estimate_cnn_confidence(image)
    final_confidence = combine_scores(feature_result.confidence, cnn_score)

    signals = list(feature_result.signals)

    if cnn_score is None:
        signals.append(
            {
                "name": "cnn_baseline_pending_deployment",
                "detail": (
                    "The baseline CNN has been started in the training "
                    "workspace, but inference weights are not wired into the "
                    "API yet."
                ),
                "score": feature_result.confidence,
            }
        )

    return DetectionPipelineResult(
        final_confidence=final_confidence,
        feature_confidence=feature_result.confidence,
        cnn_confidence=cnn_score,
        signals=signals,
        meta={
            "version": "mvp-0.3",
            "pipeline": "hybrid_feature_layer_v1",
            "deployed_cnn": cnn_score is not None,
            "feature_layer_version": "forensic_v1",
            "feature_components": feature_result.component_scores,
            "feature_vector": feature_result.vector.as_dict(),
            "raw_feature_vector": feature_result.raw_vector.as_dict(),
        },
    )
