"""End-to-end config/manifest invocation for the latency runner."""
import json
from pathlib import Path

import pytest

from experiments.latency.contracts import StructuralRunError


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


def test_build_adapters_constructs_stage_from_explicit_factory_path():
    """Ignoring explicit factory arguments would silently run the wrong stage configuration."""
    from experiments.latency.cli import build_adapters

    adapters = build_adapters({
        "stages": [{
            "name": "decode",
            "factory": "experiments.latency.adapters.video:VideoDecodeAdapter",
            "kwargs": {"max_frames": 2},
        }],
    })

    assert len(adapters) == 1
    assert adapters[0].name == "decode"
    assert adapters[0].max_frames == 2


def test_cli_writes_metadata_when_factory_construction_is_structural_failure(tmp_path):
    """Building adapters before the runner must not bypass the required audit report."""
    from experiments.latency.cli import main

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{"id": "v", "path": "video.avi"}]), encoding="utf-8")
    config = tmp_path / "latency.yaml"
    config.write_text(
        "mode: benchmark\nstages:\n  - name: i3d_extract\n"
        "    factory: experiments.latency.adapters.i3d:make_i3d_adapter\n"
        "    kwargs:\n      rgb_checkpoint: missing-rgb.pth\n      flow_checkpoint: missing-flow.pth\n"
        "      weights_mode: required\n",
        encoding="utf-8",
    )

    with pytest.raises(StructuralRunError, match="checkpoint does not exist"):
        main(["--manifest", str(manifest), "--config", str(config), "--output-dir", str(tmp_path / "out")])
    reports = list((tmp_path / "out").glob("meta_report_*.json"))
    assert len(reports) == 1


def test_cli_records_input_hashes_and_command_metadata(tmp_path):
    from experiments.latency.cli import main

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{"id": "v", "path": "video.avi"}]), encoding="utf-8")
    config = tmp_path / "latency.yaml"
    config.write_text(
        "mode: validate\nstages:\n  - name: missing\n"
        "    factory: missing.module:factory\n    kwargs: {}\n",
        encoding="utf-8",
    )

    with pytest.raises(StructuralRunError):
        main([
            "--manifest", str(manifest), "--config", str(config),
            "--mode", "validate", "--output-dir", str(tmp_path / "out"),
        ])
    report_path = next((tmp_path / "out").glob("meta_report_*.json"))
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    metadata = report["args"]["_experiment_metadata"]
    assert len(metadata["config_sha256"]) == 64
    assert len(metadata["manifest_sha256"]) == 64
    assert metadata["config_path"].endswith("latency.yaml")
