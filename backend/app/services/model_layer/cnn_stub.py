from __future__ import annotations

from functools import lru_cache

import numpy as np
from PIL import Image

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None

from app.config import get_settings


if nn is not None:
    class AEyeBaselineCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 1),
            )

        def forward(self, images: "torch.Tensor") -> "torch.Tensor":
            features = self.features(images)
            logits = self.classifier(features)
            return logits.squeeze(-1)


def get_cnn_unavailable_reason() -> str | None:
    if torch is None or nn is None:
        return "PyTorch is not installed in the backend environment."

    settings = get_settings()
    if not settings.cnn_enabled:
        return "CNN inference is disabled by the A_EYE_CNN_ENABLED setting."
    if not settings.cnn_checkpoint_path.exists():
        return f"CNN checkpoint not found at {settings.cnn_checkpoint_path}."

    return None


@lru_cache(maxsize=1)
def _load_model() -> tuple["AEyeBaselineCNN", int, dict[str, int]] | None:
    if get_cnn_unavailable_reason() is not None:
        return None

    settings = get_settings()

    checkpoint = torch.load(
        settings.cnn_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model = AEyeBaselineCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    config = checkpoint.get("config", {})
    image_size = int(config.get("image_size", 32))
    class_to_idx = checkpoint.get("class_to_idx", {"FAKE": 0, "REAL": 1})
    return model, image_size, class_to_idx


def _image_to_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    resized = image.convert("RGB").resize((image_size, image_size), resampling)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor


def estimate_cnn_confidence(image: Image.Image) -> float | None:
    loaded_model = _load_model()
    if loaded_model is None:
        return None

    model, image_size, class_to_idx = loaded_model
    with torch.no_grad():
        logits = model(_image_to_tensor(image, image_size))
        positive_probability = float(torch.sigmoid(logits).item())

    real_label = class_to_idx.get("REAL")
    fake_label = class_to_idx.get("FAKE")
    if fake_label == 1:
        fake_probability = positive_probability
    elif real_label == 1:
        fake_probability = 1.0 - positive_probability
    else:
        fake_probability = positive_probability

    return round(max(0.0, min(1.0, fake_probability)), 4)
