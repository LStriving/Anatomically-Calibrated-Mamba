from experiments.duration_error.reporting import pair_metrics


def test_pair_metrics_reports_signed_duration_and_boundary_errors():
    result = pair_metrics({"start_s": 1, "end_s": 4}, {"start_s": 2, "end_s": 4})
    assert result["duration_error_ms"] == 1000
    assert result["onset_error_ms"] == -1000
