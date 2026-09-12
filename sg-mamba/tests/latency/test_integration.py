import json

from experiments.latency.contracts import Payload
from experiments.latency.runner import STAGE_ORDER, run_benchmark


class SyntheticStage:
    def __init__(self, name):
        self.name = name

    def prepare(self, context):
        return None

    def close(self):
        return None

    def run(self, payload, context):
        values = dict(payload.values)
        values["stage_" + self.name] = True
        if self.name == "postprocess":
            values["predictions"] = {"video-id": [context["video"]["id"]]}
        return Payload(values)


def test_full_stage_contract_runs_without_checkpoints(tmp_path):
    adapters = [SyntheticStage(name) for name in STAGE_ORDER]
    report_path = run_benchmark(
        {"mode": "smoke", "weights_mode": "skip"},
        [{"id": "one", "path": "one.avi"}],
        adapters,
        tmp_path,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["scope"] == "full_pipeline"
    assert report["successes"][0]["video_id"] == "one"
    assert report["environment"]["torch"]
    assert report["repository"]["commit"]
    assert any(item["name"].startswith("predictions_") for item in report["artifacts"])