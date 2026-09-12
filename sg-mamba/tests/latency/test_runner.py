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
    assert any(item["name"].startswith("latency_summary_") for item in report["artifacts"])
    assert any(item["name"].startswith("predictions_") for item in report["artifacts"])
    summary = json.loads(next(tmp_path.glob("latency_summary_*.json")).read_text(encoding="utf-8"))
    assert summary["attempted"] == 2
    assert summary["succeeded"] == 1
    assert summary["failed"] == 1
    assert report["preparation"]["duration_ms"] >= 0
    assert report["preparation"]["stages"] == [{"stage": "decode", "status": "prepared"}]


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


class PassThrough:
    name = "decode"
    def prepare(self, context): pass
    def close(self): pass
    def run(self, payload, context): return payload.with_value("frames", [1])


class WeightedPassThrough(PassThrough):
    weights_loaded = False
    checkpoint_warnings = ["using audited test fallback"]


def test_runner_excludes_first_success_from_benchmark_aggregate(tmp_path):
    """Including the first successful video would count model warm-up as benchmark latency."""
    report_path = run_benchmark(
        {"mode": "benchmark", "weights_mode": "skip"},
        [{"id": "warmup", "path": "one.avi"}, {"id": "measured", "path": "two.avi"}],
        [PassThrough()], tmp_path,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["warmup_video_id"] == "warmup"
    assert report["aggregate_latency_ms"]["sample_count"] == 1
    assert set(report["aggregate_latency_ms"]["stages_ms"]) == {"decode"}


def test_runner_records_adapter_weight_status_and_warnings(tmp_path):
    """Dropping a degraded model's warning would make its timings look like validated inference."""
    report_path = run_benchmark(
        {"mode": "validate", "weights_mode": "skip"},
        [{"id": "v", "path": "v.avi"}], [WeightedPassThrough()], tmp_path,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["adapter_status"] == [{
        "name": "decode", "weights_loaded": False,
        "checkpoint_warnings": ["using audited test fallback"],
    }]


class CountingDecode:
    name = "decode"
    calls = 0

    def prepare(self, context): pass
    def close(self): pass
    def run(self, payload, context):
        type(self).calls += 1
        return payload.with_value("frames", [context["video"]["id"]])


def test_runner_reuses_stage_outputs_from_cache(tmp_path):
    """Leaving cache_key disconnected from runner lifecycle would execute decode twice."""
    CountingDecode.calls = 0
    config = {"mode": "validate", "cache": {"enabled": True, "dir": str(tmp_path / "cache")}}
    manifest = [{"id": "v", "path": "v.avi"}]

    run_benchmark(config, manifest, [CountingDecode()], tmp_path / "first")
    run_benchmark(config, manifest, [CountingDecode()], tmp_path / "second")

    assert CountingDecode.calls == 1
