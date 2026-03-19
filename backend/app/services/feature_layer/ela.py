from PIL import Image
import numpy as np

def quickElaScore(img: Image.Image) -> float:
    #placeholder returns a stable score for testing
    arr = np.asarray(img)
    return float((arr.std() / 128.0).clip(0.0, 1.0))
