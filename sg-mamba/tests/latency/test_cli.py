"""End-to-end config/manifest invocation for the latency runner."""
import json
from pathlib import Path

import pytest


REAL_VIDEO = Path(r"D:\LYR\lab\吞咽造影\2_2021_02_01_clip1_32s.avi")


@pytest.mark.skipif(not REAL_VIDEO.is_file(), reason="local VFSS video is unavailable")
def test_cli_runs_configured_decode_smoke_and_records_overrides(tmp_path):
    """Ignoring CLI overrides or max_frames would make the smoke run unauditable."""
    from experiments.latency.cli import main

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{"id": "vfss-smoke", "path": str(REAL_VIDEO)}]), encoding="utf-8")
    config = tmp_path / "latency.yaml"
    config.write_text("mode: benchmark\nweights_mode: required\nmax_frames: 3\nstages:\n  - decode\n", encoding="utf-8")

    report_path = main([
        "--manifest", str(manifest), "--config", str(config),
        "--mode", "smoke", "--weights-mode", "skip", "--output-dir", str(tmp_path / "out"),
    ])
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))

    assert report["args"]["mode"] == "smoke"
    assert report["args"]["weights_mode"] == "skip"
    assert report["successes"][0]["video_id"] == "vfss-smoke"
