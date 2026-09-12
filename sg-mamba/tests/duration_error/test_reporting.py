from experiments.duration_error.reporting import pair_metrics
from experiments.duration_error.analyze import run_analysis
import json
import pickle
import csv


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


def test_analysis_summary_reports_class_metrics_and_fp_fn(tmp_path):
    """A totals-only summary hides class-specific duration error and detector misses."""
    predictions = tmp_path / "result.pkl"
    with predictions.open("wb") as handle:
        pickle.dump({
            "video-id": ["v", "v", "v"],
            "t-start": [1.0, 2.0, 7.0],
            "t-end": [4.0, 5.0, 8.0],
            "score": [.9, .8, .1],
            "label": [1, 2, 1],
        }, handle)
    annotations = tmp_path / "annotations.json"
    annotations.write_text(json.dumps({"database": {"v": {
        "subset": "Test",
        "duration": 10,
        "annotations": [
            {"label": "a", "label_id": 1, "segment": [2.0, 4.0]},
            {"label": "a", "label_id": 1, "segment": [6.0, 7.0]},
            {"label": "b", "label_id": 2, "segment": [2.0, 4.0]},
        ],
    }}}), encoding="utf-8")

    run_analysis(predictions, annotations, "test", {1: "a", 2: "b"}, tmp_path, thresholds=(0.5,))
    with next(tmp_path.glob("duration_summary_*.csv")).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    row_a = next(row for row in rows if row["label_id"] == "1")
    row_b = next(row for row in rows if row["label_id"] == "2")
    assert row_a["matched_count"] == "1"
    assert row_a["false_positive_count"] == "1"
    assert row_a["false_negative_count"] == "1"
    assert row_a["duration_error_ms_mae"] == "1000.0"
    assert row_a["duration_error_ms_rmse"] == "1000.0"
    assert row_a["duration_error_ms_bias"] == "1000.0"
    assert row_a["duration_error_ms_p10"] == "1000.0"
    assert row_a["duration_error_ms_p50"] == "1000.0"
    assert row_a["duration_error_ms_p90"] == "1000.0"
    assert row_b["offset_error_ms_bias"] == "1000.0"
