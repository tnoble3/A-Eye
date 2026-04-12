# ML Workspace

This directory holds training-oriented artifacts that should evolve separately
from the lightweight backend API.

## Included Now

- `models/cnn_baseline.py`: baseline CNN architecture for early experiments
- `extract_feature_vectors.py`: shared forensic feature extraction and
  normalization pipeline for lightweight classifiers
- `requirements.txt`: training-side dependencies that are intentionally kept out
  of the API container

## Intended Next Steps

- Finalize the shortlisted dataset choice
- Build reproducible train/validation splits
- Export the best baseline weights for backend inference
