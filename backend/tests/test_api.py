from PIL import Image
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_reports_hybrid_service():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["architecture"] == "hybrid_cnn_feature"


def test_analyze_returns_stub_pipeline_response(monkeypatch):
    async def fake_fetch_image(_: str) -> Image.Image:
        return Image.new("RGB", (16, 16), color=(120, 140, 160))

    monkeypatch.setattr("app.api.routes.fetch_image", fake_fetch_image)

    response = client.post(
        "/analyze",
        json={"image_url": "https://example.com/image.png"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["meta"]["pipeline"] == "hybrid_feature_layer_v1"
    assert payload["cnn_confidence"] is None
    assert any(signal["name"] == "camera_metadata" for signal in payload["signals"])
    assert 0.0 <= payload["final_confidence"] <= 1.0
