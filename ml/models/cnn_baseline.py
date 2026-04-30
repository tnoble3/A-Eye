from __future__ import annotations

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  #back end might not have torch installed, so we provide a fallback that raises an error when the class is used
    torch = None
    nn = None


if nn is not None:
    class AEyeBaselineCNN(nn.Module):
        """Simple baseline CNN for the first training benchmark."""

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
            #The classifier stays intentionally small so the first benchmark is
            #cheap to train and easy to compare against later hybrid variants.
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
else:
    class AEyeBaselineCNN:  
        def __init__(self) -> None:
            raise RuntimeError(
                "PyTorch is required for the ML workspace. Install ml/requirements.txt "
                "before using the baseline CNN."
            )
