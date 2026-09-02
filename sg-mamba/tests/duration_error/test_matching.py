from experiments.duration_error.matching import match_at_threshold


def test_matching_claims_each_ground_truth_once_and_recalculates_thresholds():
    """Removing one-to-one matching would let two predictions claim one event."""
    predictions = [{"video_id": "v", "label_id": 1, "start_s": 0, "end_s": 10, "score": .9}, {"video_id": "v", "label_id": 1, "start_s": 2, "end_s": 8, "score": .8}]
    truth = [{"video_id": "v", "label_id": 1, "start_s": 2, "end_s": 8}]
    assert len(match_at_threshold(predictions, truth, .3)["matches"]) == 1
    assert len(match_at_threshold(predictions, truth, .7)["matches"]) == 1
