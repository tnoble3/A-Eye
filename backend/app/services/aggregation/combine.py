def combine_scores(feature_score: float, cnn_score: float | None) -> float:
    # The hybrid architecture treats the CNN as the stronger signal once it is
    # deployed, but the API still needs a stable fallback while development is
    # in the stub stage.
    #the cnn is treated as a stronger singal becuase it is trained on the dataset itself
    #the current cnn is 91.5% accurate on the test set, while the feature score is around 80% accurate on the same set. I will retrain with 
    #a more powerful CPUN and more epochs to see if I can get the CNN up to 95%+ accuracy, which should make it a much stronger signal in the hybrid score.
    if cnn_score is None:
        return feature_score
    return round((0.4 * feature_score) + (0.6 * cnn_score), 4)
