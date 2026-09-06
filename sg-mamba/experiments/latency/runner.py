"""Stage-ordered, failure-isolated latency execution."""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns
from statistics import mean

from .contracts import Payload, RecoverableVideoError, StructuralRunError
from .metadata import write_meta_report


STAGE_ORDER = ("decode", "flow_extract", "i3d_extract", "coarse_detector", "keypoint_extract", "kalman_smooth", "skeleton_encode", "fine_detector", "postprocess")


def run_benchmark(config, manifest, adapters, output_dir):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    adapters = list(adapters)
    names = [adapter.name for adapter in adapters]
    if names != [name for name in STAGE_ORDER if name in names]:
        raise StructuralRunError("Adapters must be supplied in pipeline order")
    successes, failures, rows, predictions = [], [], [], []
    preparation_started = perf_counter_ns()
    preparation = []
    try:
        for adapter in adapters:
            adapter.prepare(config)
            preparation.append({"stage": adapter.name, "status": "prepared"})
        preparation_ms = (perf_counter_ns() - preparation_started) / 1_000_000
        for video in manifest:
            payload, timings, stage = Payload({"video": video}), {}, None
            try:
                _reset_peak_gpu_memory()
                for adapter in adapters:
                    stage = adapter.name; _cuda_synchronize(); start = perf_counter_ns()
                    payload = adapter.run(payload, {"config": config, "video": video})
                    _cuda_synchronize()
                    timings[stage + "_ms"] = (perf_counter_ns() - start) / 1_000_000
                peak_gpu_memory = _peak_gpu_memory()
                total_ms = sum(timings.values())
                successes.append({"video_id": video["id"], "stage_timings_ms": timings,
                                  "peak_gpu_memory_bytes": peak_gpu_memory})
                rows.append({"video_id": video["id"], "input_path": video["path"],
                             "status": "success", "total_ms": total_ms,
                             "peak_gpu_memory_bytes": peak_gpu_memory, **timings})
                if "predictions" in payload.values:
                    predictions.append({"video_id": video["id"],
                                        "predictions": _json_safe(payload.require("predictions"))})
            except RecoverableVideoError as error:
                failures.append({"video_id": video["id"], "failure_stage": stage, "failure_message": str(error)})
                rows.append({"video_id": video["id"], "input_path": video["path"],
                             "status": "failed", "failure_stage": stage,
                             "failure_message": str(error)})
    except StructuralRunError as error:
        write_meta_report(output_dir, {
            "args": config,
            "preparation": {"duration_ms": (perf_counter_ns() - preparation_started) / 1_000_000,
                            "stages": preparation},
            "scope": "validation" if config.get("mode") == "validate" else "partial_pipeline",
            "successes": successes,
            "failures": failures,
            "structural_failure": {"message": str(error), "type": type(error).__name__},
            "adapter_status": _adapter_status(adapters),
            "artifacts": [],
        })
        raise
    finally:
        for adapter in reversed(adapters): adapter.close()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results = output_dir / ("latency_results_" + stamp + ".csv")
    summary_path = output_dir / ("latency_summary_" + stamp + ".json")
    predictions_path = output_dir / ("predictions_" + stamp + ".json")
    fields = sorted({key for row in rows for key in row})
    with results.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    warmup_video_id, aggregate = _aggregate_successes(successes, config.get("mode"))
    summary = {
        "attempted": len(manifest),
        "succeeded": len(successes),
        "failed": len(failures),
        "warmup_video_id": warmup_video_id,
        "aggregate_latency_ms": aggregate,
        "failure_reasons": failures,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    predictions_path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    scope = _result_scope(config, names, failures)
    return write_meta_report(output_dir, {
        "args": config,
        "preparation": {"duration_ms": preparation_ms, "stages": preparation},
        "scope": scope,
        "successes": successes,
        "failures": failures,
        "warmup_video_id": warmup_video_id,
        "aggregate_latency_ms": aggregate,
        "adapter_status": _adapter_status(adapters),
        "artifacts": [{"path": results, "purpose": "per-video latency results"},
                      {"path": summary_path, "purpose": "latency summary"},
                      {"path": predictions_path, "purpose": "video predictions"}],
    })


def _result_scope(config, names, failures):
    if config.get("mode") == "validate":
        return "validation"
    if set(names) == set(STAGE_ORDER) and not failures:
        return "full_pipeline"
    return "partial_pipeline"


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        return value.item()
    return value


def _aggregate_successes(successes, mode):
    measured = successes[1:] if mode == "benchmark" else successes
    warmup_video_id = successes[0]["video_id"] if mode == "benchmark" and successes else None
    stage_names = sorted({name for success in measured for name in success["stage_timings_ms"]})
    stages = {
        name.removesuffix("_ms"): mean(success["stage_timings_ms"][name] for success in measured if name in success["stage_timings_ms"])
        for name in stage_names
    }
    return warmup_video_id, {"sample_count": len(measured), "stages_ms": stages}


def _adapter_status(adapters):
    """Expose checkpoint state without treating a missing attribute as a claim."""
    status = []
    for adapter in adapters:
        if not hasattr(adapter, "weights_loaded") and not hasattr(adapter, "checkpoint_warnings"):
            continue
        status.append({
            "name": adapter.name,
            "weights_loaded": getattr(adapter, "weights_loaded", None),
            "checkpoint_warnings": list(getattr(adapter, "checkpoint_warnings", [])),
        })
    return status


def _cuda_synchronize():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except (ImportError, RuntimeError):
        return None


def _reset_peak_gpu_memory():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except (ImportError, RuntimeError):
        return None


def _peak_gpu_memory():
    try:
        import torch
        if torch.cuda.is_available():
            return int(torch.cuda.max_memory_allocated())
    except (ImportError, RuntimeError):
        return None
    return None
