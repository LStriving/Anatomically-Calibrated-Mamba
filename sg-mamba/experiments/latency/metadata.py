"""Timestamped metadata reports that index all generated experiment output."""
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Union


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


def write_meta_report(output_dir: Union[str, Path], metadata: Mapping[str, Any]) -> Path:
    """Write the authoritative artifact index, adding hashes for extant files."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    report = dict(metadata)
    report.setdefault("timestamp_utc", datetime.now(timezone.utc).isoformat())
    report.setdefault("environment", {
        "python": sys.version,
        "platform": platform.platform(),
    })
    indexed = []
    for item in report.get("artifacts", []):
        entry = dict(item)
        path = Path(entry.pop("path"))
        entry["name"] = entry.get("name", path.name)
        entry["sha256"] = sha256_file(path) if path.is_file() else None
        indexed.append(entry)
    report["artifacts"] = indexed
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = destination / "meta_report_{}.json".format(timestamp)
    suffix = 1
    while report_path.exists():
        report_path = destination / "meta_report_{}_{}.json".format(timestamp, suffix)
        suffix += 1
    report_path.write_text(json.dumps(report, indent=2, default=_json_value), encoding="utf-8")
    return report_path
