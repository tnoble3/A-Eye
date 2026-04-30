# Model Evaluation Focus

The current evaluation plan emphasizes metrics that fit the browser safety use
case and the hybrid detection pipeline.

## Priority Metrics

- F1 score: balances precision and recall when the classes are not perfectly
  balanced.
- Precision: important because false alarms can erode trust in the product.
- False positive rate: tracked explicitly because a detector that flags too many
  authentic images is hard to deploy in a real browsing workflow.

## Why These Metrics Matter

The extension is meant to provide quick decision support, not just benchmark
accuracy. That means the project needs to care about the user cost of a wrong
flag as much as raw classification performance.

