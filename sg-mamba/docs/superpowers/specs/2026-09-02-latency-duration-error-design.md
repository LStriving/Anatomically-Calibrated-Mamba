# Latency and Duration Error Experiment Design

## Purpose and scope

This design implements the two P0 experiments in `EXPERIMENT_PLANS.md` without changing the existing model evaluation entry points. It has two independent deliverables:

1. An end-to-end latency benchmark from an untrimmed VFSS video to predicted micro-action onset/offsets.
2. A duration-error analysis that compares saved predictions with test-set ground truth.

The latency benchmark follows the planned full pipeline: video decode, coarse detector, optical-flow extraction, I3D appearance extraction, keypoint extraction, Kalman smoothing, skeleton encoding, fine detection, and post-processing. It must report the status and measured cost of every available stage; it must never substitute estimates for unimplemented stages.

The existing `sg-mamba` evaluators operate on pre-extracted `.npy` features and `eval2tower.py` can write `final_result.pkl` with `video-id`, `t-start`, `t-end`, `score`, and `label`. The duration analysis consumes this result format and the annotation JSON independently of Mamba, so it can run and be tested on Windows.

## Non-goals

- Do not change model architecture, training, labels, or the existing evaluator's primary behavior.
- Do not treat validation-mode, random-weight, partial-pipeline, or failed-run output as rebuttal evidence without human review.
- Do not include model construction, checkpoint I/O, dependency startup, or warm-up in per-video inference latency.
- Do not claim that duration error replaces mAP.

## Latency architecture

Create a standalone `experiments/latency/` package. A runner owns video iteration, cache lifecycle, common timing, report generation, and the run manifest. Each pipeline step is an adapter with a narrow protocol:

```python
prepare(context) -> None
run(payload, context) -> payload
close() -> None
```

`prepare` constructs the model, optionally loads a checkpoint, and runs warm-up; none of this is per-video latency. `run` processes exactly one video and returns a new, named payload. `close` releases stage resources. Payloads expose only stage-relevant named values such as `frames`, `flow`, `rgb_flow_features`, `keypoints`, `proposals`, and `predictions`; no stage may rely on an upstream private temporary path.

The runner invokes the following ordered adapters:

```text
video_decode -> coarse_detector -> flow_extract -> i3d_extract
-> keypoint_extract -> kalman_smooth -> skeleton_encode
-> fine_detector -> postprocess -> predictions
```

The cache manager owns disk-backed intermediate values. It keys entries by `run_id`, `video_id`, stage, input hash, configuration hash, and stage version. It deletes an intermediate after its final declared consumer unless the run requests retention for debugging.

### Weight modes and execution modes

The runner exposes three execution modes:

- `validate`: local contract validation. It permits `weights_mode=skip` and may use test adapters or randomly initialized models. Every output is labelled `non_benchmark`.
- `smoke`: one or two videos on the target machine. It validates installed dependencies and stage connections and may skip checkpoints. It is not a formal benchmark.
- `benchmark`: the full selected manifest and real adapters. Checkpoints are normally preloaded and warmed up before timing.

`weights_mode` is explicit:

- `required`: attempt to load each requested checkpoint during `prepare`.
- `skip`: construct the model but deliberately do not load a checkpoint, allowing real tensor/operator paths to be tested locally.

A missing or unloadable checkpoint is a warning, not a fatal error. The runner continues with the random-initialized model if that adapter can run, records `weights_loaded=false`, and records the requested path and warning in the metadata. A missing real adapter, an invalid input schema, or an incompatible stage interface is structural and stops the run because the resulting pipeline is not interpretable.

### Timing and memory contract

Measure CPU wall time with `time.perf_counter_ns()`. Surround every GPU stage boundary with `torch.cuda.synchronize()` before sampling the end timestamp. For each video, reset peak GPU memory statistics before the first stage and record `torch.cuda.max_memory_allocated()` after the last stage. The first complete video is warm-up and is excluded from formal aggregate latency statistics.

Model construction, checkpoint loading, and warm-up are reported separately as environment-preparation metadata. The per-video latency includes all video-to-prediction stages, including decoding and post-processing.

## Latency inputs and outputs

The input is a fixed JSON video manifest with a video identifier, source path, split, expected duration where available, and optional integrity hash. The configuration identifies stages, adapter settings, devices, cache policy, retention policy, and requested checkpoints.

Each run writes timestamped artifacts:

```text
latency_results_<timestamp>.csv
latency_summary_<timestamp>.json
predictions_<timestamp>.json
meta_report_<timestamp>.json
```

`latency_results` has one row per attempted video and includes:

```text
run_id, video_id, input_path, video_duration_s, num_frames,
execution_mode, weights_mode, weights_loaded, device, stage_status,
decode_ms, coarse_ms, flow_ms, i3d_ms, keypoint_ms, kalman_ms,
skeleton_encode_ms, fine_ms, postprocess_ms, total_ms,
peak_gpu_memory_mb, failure_stage, failure_message
```

The summary states attempted, succeeded, and failed counts; all aggregate values are calculated over successful rows only. A recoverable per-video error never suppresses artifacts from successful videos. The summary must identify the reduced cohort and failure reasons.

## Duration-error architecture

Create an independent `experiments/duration_error/` package. It reads an existing `final_result.pkl` and an annotation JSON and normalizes them to two internal tables:

```text
Prediction: video_id, label_id, label_name, start_s, end_s, score
Ground truth: video_id, label_id, label_name, start_s, end_s
```

The loader validates the result keys, requested split, label-ID/name mapping, and time bounds `0 <= start < end <= video_duration`. A schema, split, or label mismatch is fatal rather than silently paired.

For every `(video_id, class_name)` group, compute all same-class temporal IoUs and greedily select non-conflicting pairs from highest IoU downward. Run this matching separately at tIoU thresholds `0.3`, `0.5`, and `0.7`; do not reuse lower-threshold pairs at a stricter threshold. Unmatched predictions and ground-truth instances are respectively recorded as FP and FN and are excluded from duration-error values.

For each matched pair:

```text
duration_error_ms = ((pred_end - pred_start) - (gt_end - gt_start)) * 1000
relative_error_pct = 100 * duration_error_ms / duration_gt_ms
onset_error_ms = (pred_start - gt_start) * 1000
offset_error_ms = (pred_end - gt_end) * 1000
```

Relative error is only defined for a strictly positive ground-truth duration. The report records any excluded invalid or zero-duration annotations rather than silently discarding them.

Outputs are timestamped and linked in the metadata report:

```text
duration_pairs_<timestamp>.csv
duration_summary_<timestamp>.csv
duration_limitations_<timestamp>.md
meta_report_<timestamp>.json
```

The pair table includes all values above plus tIoU and score. The summary is stratified by tIoU threshold and class, with matched count, GT count, prediction count, match rate, mean GT duration, duration MAE, RMSE, signed bias, onset bias, offset bias, and p10/median/p90 signed duration error.

## Relation to mAP and rebuttal limits

Duration Error measures boundary and duration agreement only for class-consistent matched instances. It does not measure missed detections, false positives, or incorrect classes. It must therefore be reported beside mAP, not as a replacement.

mAP evaluates precision-recall after ranking predictions by confidence. At each tIoU threshold it reflects confidence ordering as well as the effects of classification, localization, false positives, and false negatives. The duration report must show results at tIoU 0.3/0.5/0.7 and the matched count for each threshold, because stricter thresholds restrict analysis to a smaller, typically easier-to-localize subset.

`duration_limitations_<timestamp>.md` is a factual template, not an automatic scientific conclusion. It must state that measurements are conditional on the matching threshold, are agreement with existing ground truth rather than clinical absolute truth, and cannot supersede mAP. Rebuttal prose may use only actual reported values after manual review of the associated metadata.

## Metadata and auditability

Every latency or duration run creates a `meta_report_<timestamp>.json` as the authoritative artifact index. It contains:

- `run_id`, timestamp, repository commit, Python/PyTorch/CUDA versions, GPU and CPU identity;
- full command-line parameters, config path and hash, input manifest and input hashes;
- execution mode, requested checkpoint paths, checkpoint load status, weight warnings, adapter availability, adapter versions, and per-stage timing summaries;
- attempted/succeeded/failed video lists and failure information;
- `result_scope` of `full_pipeline`, `partial_pipeline`, or `validation`;
- each generated artifact filename, timestamp, purpose, and content hash.

This report is produced even when a run has no successful videos, so the failure remains auditable. No output is suppressed merely because checkpoints are absent or a subset of videos fails; outputs must instead disclose their conditions and cohort.

## Verification strategy

### Local Windows verification

- Run `validate` with skipped weights to test adapter order, payload schemas, cache invalidation, timestamped artifacts, and metadata generation.
- Run real optical-flow and I3D smoke tests when their local dependencies and checkpoints are available.
- Use synthetic prediction PKL and annotation JSON fixtures to assert one-to-one matching, all three tIoU thresholds, FP/FN accounting, MAE/RMSE/bias calculations, and metadata references.
- Do not use local validation values as paper or rebuttal results.

### Cloud benchmark verification

- Use a fixed test video manifest, recorded command, environment, configuration, and checkpoint identifiers.
- Preload real checkpoints and warm up before formal timing, while preserving separate preparation metadata.
- Preserve successful partial results if recoverable failures occur; disclose their counts and reasons in the summary and metadata report.
- Review the associated metadata before using any result in the manuscript or rebuttal.

## Acceptance criteria

1. A latency run produces per-video records, a summary, predictions, and an authoritative metadata report with every generated filename and hash.
2. A missing checkpoint produces a warning and explicit `weights_loaded=false`, not a silent fallback or fatal error.
3. A structural adapter/schema failure stops the run with an auditable metadata report.
4. Recoverable per-video failures do not discard successful-video artifacts or summaries.
5. Duration analysis reads the existing evaluator's PKL schema, validates annotation compatibility, and reports independent 0.3/0.5/0.7 matching results.
6. Duration artifacts explicitly distinguish conditional boundary agreement from confidence-aware mAP.
