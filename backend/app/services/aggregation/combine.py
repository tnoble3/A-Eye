def combine_scores(feature_score: float, cnn_score: float | None) -> float:
    if cnn_score is None:
        return feature_score
    return 0.4 * feature_score + 0.6 * cnn_score