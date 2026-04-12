from PIL import Image


def estimate_cnn_confidence(_: Image.Image) -> float | None:
    # The baseline CNN now lives in the training workspace, but the backend API
    # keeps returning a stable response shape until exported inference weights
    # are packaged for deployment.
    return None
