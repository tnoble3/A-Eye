# Model Training Notes

## Current Status

- A baseline CNN has been added in the `ml/` workspace.
- Training strategy research is still in progress.
- Final dataset selection is intentionally deferred until the short-listed
  options are compared on quality, licensing, and domain fit.

## Dataset Comparison Framework

The project is currently comparing a small number of candidate image datasets.
Each candidate should be reviewed against the same criteria:

- Source credibility and licensing
- Balance between authentic and AI-generated images
- Diversity of generators, subjects, and compression artifacts
- Suitability for browser-sourced images
- Ease of preprocessing into a reproducible training split

## Training Direction

- Start with the baseline CNN to establish a measurable benchmark.
- Add feature-level signals later to test whether the hybrid approach improves
  performance and explainability.
- Preserve a clear separation between training code in `ml/` and inference code
  in `backend/` so deployment stays lightweight.
