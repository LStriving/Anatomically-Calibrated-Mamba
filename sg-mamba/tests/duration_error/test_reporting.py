from experiments.duration_error.reporting import pair_metrics
from experiments.duration_error.analyze import run_analysis
import json
import pickle


def test_pair_metrics_reports_signed_duration_and_boundary_errors():
    result = pair_metrics({"start_s": 1, "end_s": 4}, {"start_s": 2, "end_s": 4})
    assert result["duration_error_ms"] == 1000
    assert result["onset_error_ms"] == -1000


def test_analysis_indexes_pairs_summary_and_factual_limitations(tmp_path):
    predictions = tmp_path / "result.pkl"
    with predictions.open("wb") as handle:
        pickle.dump({"video-id": ["v"], "t-start": [1], "t-end": [4], "score": [.9], "label": [1]}, handle)
    annotations = tmp_path / "annotations.json"
    annotations.write_text('{"database":{"v":{"subset":"Test","duration":5,"annotations":[{"label":"swallow","label_id":1,"segment":[2,4]}]}}}', encoding="utf-8")
    report = json.loads(run_analysis(predictions, annotations, "test", {1: "swallow"}, tmp_path).read_text())
    names = {artifact["name"] for artifact in report["artifacts"]}
    assert any(name.startswith("duration_pairs_") for name in names)
    assert any(name.startswith("duration_summary_") for name in names)
    assert any(name.startswith("duration_limitations_") for name in names)
