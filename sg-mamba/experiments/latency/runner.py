"""Stage-ordered, failure-isolated latency execution."""
import csv
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns

from .contracts import Payload, RecoverableVideoError, StructuralRunError
from .metadata import write_meta_report


STAGE_ORDER = ("decode", "coarse_detector", "flow_extract", "i3d_extract", "keypoint_extract", "kalman_smooth", "skeleton_encode", "fine_detector", "postprocess")


def run_benchmark(config, manifest, adapters, output_dir):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    adapters = list(adapters)
    names = [adapter.name for adapter in adapters]
    if names != [name for name in STAGE_ORDER if name in names]:
        raise StructuralRunError("Adapters must be supplied in pipeline order")
    successes, failures, rows = [], [], []
    try:
        for adapter in adapters: adapter.prepare(config)
        for video in manifest:
            payload, timings, stage = Payload({"video": video}), {}, None
            try:
                for adapter in adapters:
                    stage = adapter.name; start = perf_counter_ns()
                    payload = adapter.run(payload, {"config": config, "video": video})
                    timings[stage + "_ms"] = (perf_counter_ns() - start) / 1_000_000
                successes.append({"video_id": video["id"], "stage_timings_ms": timings})
                rows.append({"video_id": video["id"], "status": "success", **timings})
            except RecoverableVideoError as error:
                failures.append({"video_id": video["id"], "failure_stage": stage, "failure_message": str(error)})
                rows.append({"video_id": video["id"], "status": "failed", "failure_stage": stage, "failure_message": str(error)})
    except StructuralRunError as error:
        write_meta_report(output_dir, {
            "args": config,
            "scope": "validation" if config.get("mode") == "validate" else "partial_pipeline",
            "successes": successes,
            "failures": failures,
            "structural_failure": {"message": str(error), "type": type(error).__name__},
            "artifacts": [],
        })
        raise
    finally:
        for adapter in reversed(adapters): adapter.close()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results = output_dir / ("latency_results_" + stamp + ".csv")
    fields = sorted({key for row in rows for key in row})
    with results.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    return write_meta_report(output_dir, {"args": config, "scope": "validation" if config.get("mode") == "validate" else "partial_pipeline", "successes": successes, "failures": failures, "artifacts": [{"path": results, "purpose": "latency results"}]})
