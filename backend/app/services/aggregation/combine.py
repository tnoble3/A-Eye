def combine_scores(feature_score: float, cnn_score: float | None) -> float:
    #the cnn is around 91% accurate, trained on 15 epochs
    #feature level layer is around 80% accurate
    if cnn_score is None:
        return feature_score
    return round((0.3 * feature_score) + (0.7 * cnn_score), 4)
