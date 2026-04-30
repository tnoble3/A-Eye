from PIL import Image

from app.services.aggregation import combine_scores
from app.services.feature_layer import analyze_feature_layer, extract_feature_vector_bundle
from app.services.pipeline import run_detection_pipeline


def test_combine_scores_falls_back_to_feature_score():
    assert combine_scores(0.42, None) == 0.42


def test_combine_scores_uses_weighted_hybrid_when_cnn_available():
    assert combine_scores(0.20, 0.80) == 0.62


def test_feature_layer_returns_numeric_vector_and_interpretable_signals():
    result = analyze_feature_layer(Image.new("RGB", (16, 16), color=(20, 40, 60)))

    assert 0.0 <= result.confidence <= 1.0
    assert set(result.raw_vector.as_dict()) == set(result.vector.as_dict())
    assert all(0.0 <= value <= 1.0 for value in result.vector.as_dict().values())
    assert result.vector.as_dict()["metadata_missing_camera_data"] == 1.0
    assert any(signal["name"] == "camera_metadata" for signal in result.signals)
    assert any(signal["name"] == "compression_anomalies" for signal in result.signals)


def test_feature_vector_bundle_exposes_raw_and_normalized_vectors():
    bundle = extract_feature_vector_bundle(Image.new("RGB", (16, 16), color=(80, 120, 200)))
    raw_values = bundle.raw.as_dict()

    assert set(bundle.raw.as_dict()) == set(bundle.normalized.as_dict())
    assert "ela_p95_residual" in raw_values
    assert "ela_p99_residual" in raw_values
    assert raw_values["ela_p99_residual"] >= raw_values["ela_p95_residual"]
    assert any(value > 0.0 for value in raw_values.values())
    assert all(0.0 <= value <= 1.0 for value in bundle.normalized.as_dict().values())


def test_pipeline_marks_cnn_as_unavailable(monkeypatch):
    monkeypatch.setattr("app.services.pipeline.estimate_cnn_confidence", lambda _: None)
    monkeypatch.setattr(
        "app.services.pipeline.get_cnn_unavailable_reason",
        lambda: "CNN checkpoint not found at /app/ml/artifacts/cnn_baseline/best_model.pt.",
    )
    result = run_detection_pipeline(Image.new("RGB", (8, 8), color=(20, 40, 60)))

    assert result.cnn_confidence is None
    assert result.meta["deployed_cnn"] is False
    assert result.meta["pipeline"] == "hybrid_feature_layer_v1"
    assert "feature_vector" in result.meta
    assert "raw_feature_vector" in result.meta
    assert any(signal["name"] == "cnn_baseline_unavailable" for signal in result.signals)
    assert any(
        signal["detail"] == "CNN checkpoint not found at /app/ml/artifacts/cnn_baseline/best_model.pt."
        for signal in result.signals
    )


def test_pipeline_uses_deployed_cnn_score(monkeypatch):
    monkeypatch.setattr("app.services.pipeline.estimate_cnn_confidence", lambda _: 0.80)
    result = run_detection_pipeline(Image.new("RGB", (8, 8), color=(20, 40, 60)))

    assert result.cnn_confidence == 0.80
    assert result.meta["deployed_cnn"] is True
    assert result.final_confidence == combine_scores(result.feature_confidence, 0.80)
    assert not any(signal["name"] == "cnn_baseline_unavailable" for signal in result.signals)
