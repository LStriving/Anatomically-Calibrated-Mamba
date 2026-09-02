import csv
import hashlib
import json

import pytest

from experiments.latency.contracts import Payload, StructuralRunError
from experiments.latency.metadata import write_meta_report


def test_payload_requires_declared_value():
    """Removing payload schema validation must reject missing stage inputs."""
    with pytest.raises(StructuralRunError, match="frames"):
        Payload({}).require("frames")


def test_meta_report_hashes_existing_artifacts(tmp_path):
    """Removing artifact hashing must make audit metadata incomplete."""
    artifact = tmp_path / "latency_results.csv"
    artifact.write_text("video_id,total_ms\ngood,10\n", encoding="utf-8")

    report_path = write_meta_report(
        tmp_path,
        {"artifacts": [{"path": str(artifact), "purpose": "latency results"}]},
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert report["artifacts"][0]["sha256"] == expected_hash
    assert len(report["artifacts"][0]["sha256"]) == 64
