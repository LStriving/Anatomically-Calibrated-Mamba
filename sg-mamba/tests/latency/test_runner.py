import json
from pathlib import Path

import pytest
from experiments.latency.contracts import Payload, RecoverableVideoError
from experiments.latency.contracts import StructuralRunError
from experiments.latency.runner import run_benchmark


class Decode:
    name = "decode"
    def prepare(self, context): pass
    def close(self): pass
    def run(self, payload, context):
        if context["video"]["id"] == "bad":
            raise RecoverableVideoError("unreadable")
        return payload.with_value("frames", [1])


def test_runner_preserves_success_when_one_video_is_recoverable(tmp_path):
    """Removing per-video isolation must discard the good latency artifact."""
    report_path = run_benchmark(
        {"mode": "validate", "weights_mode": "skip"},
        [{"id": "good", "path": "good.mp4"}, {"id": "bad", "path": "bad.mp4"}],
        [Decode()], tmp_path,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert [item["video_id"] for item in report["successes"]] == ["good"]
    assert [item["video_id"] for item in report["failures"]] == ["bad"]
    assert any(item["name"].startswith("latency_results_") for item in report["artifacts"])


class BrokenPrepare:
    name = "decode"
    def prepare(self, context): raise StructuralRunError("factory unavailable")
    def close(self): pass


def test_runner_writes_audit_report_before_reraising_structural_error(tmp_path):
    """Removing structural failure reporting would leave unavailable adapters unauditable."""
    with pytest.raises(StructuralRunError, match="factory unavailable"):
        run_benchmark({"mode": "validate"}, [{"id": "v", "path": "v.avi"}], [BrokenPrepare()], tmp_path)
    reports = list(Path(tmp_path).glob("meta_report_*.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert report["structural_failure"]["message"] == "factory unavailable"
