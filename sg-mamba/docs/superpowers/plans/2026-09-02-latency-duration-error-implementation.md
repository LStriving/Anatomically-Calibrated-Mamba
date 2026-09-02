# Latency and Duration Error Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement an auditable raw-video latency runner and independent duration-error CLI.

**Architecture:** `experiments.latency` owns stage contracts, timing, cache, artifact output, and metadata. `experiments.duration_error` reads the existing evaluator PKL and annotations without importing Mamba.

**Tech Stack:** Python 3, standard library, NumPy, PyTorch, OpenCV when installed, pytest.

## Global Constraints

- Preserve existing evaluator behavior.
- Exclude model construction, checkpoint I/O, and warm-up from per-video latency.
- Missing checkpoint is a warning; missing adapter/schema mismatch/split mismatch is structural.
- Preserve successful partial results and index all outputs in `meta_report_<timestamp>.json`.
- Duration Error is matched-boundary agreement; mAP also measures confidence-ranked FP/FN, class, and localization performance.

### Task 1: Define shared latency contracts and metadata

**Files:** Create `experiments/__init__.py`, `experiments/latency/{__init__,contracts,metadata}.py`, and `tests/latency/test_contracts.py`.

**Interfaces:** `Payload.require(name)`, `Payload.with_value(name, value)`, `StageAdapter.prepare/run/close`, `StructuralRunError`, `RecoverableVideoError`, `write_meta_report(output_dir, metadata)`.

- [ ] Write a failing test: `Payload({}).require("frames")` raises `StructuralRunError`; a metadata report of a temporary CSV has a 64-character SHA-256.
- [ ] Run `python -m pytest tests/latency/test_contracts.py -v`; expect import failure.
- [ ] Implement serializable stage/video records; exact enums are `validate|smoke|benchmark`, `skip|required`, `validation|partial_pipeline|full_pipeline`. Metadata includes args, hashes, environment, checkpoint warnings/status, successes/failures, scope, artifact name/hash.
- [ ] Run the same test; expect PASS.
- [ ] Commit: `git add experiments tests/latency/test_contracts.py; git commit -m "feat: define experiment contracts and metadata"`.

### Task 2: Implement latency runner and adapters

**Files:** Create `experiments/latency/{timing,cache,runner,cli}.py`, `experiments/latency/adapters/{__init__,video,flow,model}.py`, and `tests/latency/test_runner.py`.

**Interfaces:** `run_benchmark(config, manifest, adapters, output_dir) -> Path`; CLI flags are `--manifest --config --mode --weights-mode --output-dir`.

- [ ] Write a failing test using a decode adapter and an adapter that raises `RecoverableVideoError` only for video `bad`; assert metadata lists `good` successful, `bad` failed, and has a `latency_results_` artifact.
- [ ] Run `python -m pytest tests/latency/test_runner.py -v`; expect import failure.
- [ ] Implement ordered stages `decode, coarse_detector, flow_extract, i3d_extract, keypoint_extract, kalman_smooth, skeleton_encode, fine_detector, postprocess`; use `perf_counter_ns` and CUDA synchronization, reset/read peak GPU memory per video, and exclude first success from aggregates as warm-up. Structural failure stops but writes metadata; per-video failure continues. Cache key hashes run/video/stage/config/input/version. Video adapter uses `cv2.VideoCapture`; flow adapter calls `DualTVL1OpticalFlow_create` without global path effects; model adapter warns and uses random weights after checkpoint-load failure but fails for unavailable factory/import.
- [ ] Run `python -m pytest tests/latency -v`; expect PASS, with unavailable OpenCV tests using `pytest.importorskip("cv2")`.
- [ ] Commit: `git add experiments/latency tests/latency; git commit -m "feat: add latency pipeline runner"`.

### Task 3: Implement duration schema and matching

**Files:** Create `experiments/duration_error/{__init__,schema,matching}.py`, `tests/duration_error/test_schema.py`, and `tests/duration_error/test_matching.py`.

**Interfaces:** `load_predictions(path, label_map)`, `load_ground_truth(path, split, label_map)`, `match_at_threshold(predictions, ground_truth, threshold)`.

- [ ] Write failing tests: PKL missing `t-start` raises `ValueError`; two same-video/class predictions can only match one GT; matching at 0.3 and 0.7 is independently recalculated.
- [ ] Run `python -m pytest tests/duration_error/test_schema.py tests/duration_error/test_matching.py -v`; expect import failure.
- [ ] Require equal-length `video-id/t-start/t-end/score/label` arrays, valid positive segments, selected split, known labels, and declared bounds. Group by video/class, sort candidates `(-tIoU, -score, start times)`, and greedily claim unclaimed pairs.
- [ ] Run the same tests; expect PASS.
- [ ] Commit: `git add experiments/duration_error tests/duration_error; git commit -m "feat: load and match duration inputs"`.

### Task 4: Implement duration metrics, outputs, and CLI

**Files:** Create `experiments/duration_error/{reporting,analyze,cli}.py`, `tests/duration_error/{test_reporting,test_cli}.py`, and `configs/latency_benchmark.example.yaml`; modify the approved design spec with exact invocations.

**Interfaces:** `run_analysis(predictions, annotations, split, label_map, output_dir, thresholds) -> Path`; CLI accepts `--predictions --annotations --split --label-map --tiou-thresholds --output-dir`.

- [ ] Write failing tests: prediction [1,4] versus GT [2,4] has `duration_error_ms=1000` and `onset_error_ms=-1000`; a completed analysis has rows for 0.3/0.5/0.7 plus limitations and metadata artifacts; tIoU=1.0 is rejected.
- [ ] Run `python -m pytest tests/duration_error/test_reporting.py tests/duration_error/test_cli.py -v`; expect import failure.
- [ ] Implement pair CSV, class/threshold summary CSV, factual limitations markdown, and hashed metadata. Report signed duration/onset/offset errors, MAE, RMSE, bias, p10/p50/p90, matched/GT/prediction counts, and match rate. Relative error exists only for positive GT duration. Limitations explicitly state that mAP ranks confidence and includes FP/FN, classification, and localization. The example config lists all nine latency stages and explicit factory import paths.
- [ ] Run `python -m pytest tests/latency tests/duration_error -v`; expect PASS except explicitly skipped optional OpenCV tests.
- [ ] Commit: `git add experiments configs tests docs/superpowers/specs/2026-09-02-latency-duration-error-design.md; git commit -m "feat: add duration error reporting"`.
