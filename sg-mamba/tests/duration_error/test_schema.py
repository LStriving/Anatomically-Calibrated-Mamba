import pickle

import pytest

from experiments.duration_error.schema import load_predictions, load_ground_truth


def test_prediction_loader_rejects_missing_required_array(tmp_path):
    """Removing evaluator-schema validation accepts unusable result PKLs."""
    path = tmp_path / "result.pkl"
    with path.open("wb") as handle:
        pickle.dump({"video-id": ["v"], "t-end": [1], "score": [.8], "label": [1]}, handle)
    with pytest.raises(ValueError, match="t-start"):
        load_predictions(path, {1: "swallow"})


def test_ground_truth_loader_selects_requested_split(tmp_path):
    path = tmp_path / "annotations.json"
    path.write_text('{"database":{"v":{"subset":"Test","duration":5,"annotations":[{"label":"swallow","label_id":1,"segment":[1,2]}]}}}', encoding="utf-8")
    assert load_ground_truth(path, "test", {1: "swallow"})[0]["video_id"] == "v"
