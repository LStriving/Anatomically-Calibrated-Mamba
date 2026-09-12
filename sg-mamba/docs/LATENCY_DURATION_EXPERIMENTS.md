# Latency and Duration Experiments

本文档说明 `codex/latency-duration-experiments` 分支当前完成的功能、尚未完成的工作，以及 Windows + Conda `pytorch` 环境中的使用方法。

## 当前状态

### 已完成

- 保持原有 `eval2tower.py` 不变。
- 建立 latency stage contract：`Payload`、`prepare/run/close`、结构性错误与单视频可恢复错误。
- latency runner 支持 `validate`、`smoke`、`benchmark` 三种模式。
- 每个视频独立计时和失败隔离，使用 `perf_counter_ns` 计时，并记录 CUDA 同步和峰值显存。
- benchmark 模式排除第一个成功视频作为 warm-up。
- 模型构造、checkpoint 加载和 adapter preparation 位于视频计时区间之外。
- 已接入 raw-video stage：decode、TV-L1 flow、RGB/flow I3D、keypoint、Kalman、skeleton/heatmap、coarse detector、eval2tower fine detector 和 postprocess。
- 粗检测输入为 evaluator 的 `list[dict]`；双塔细检测输入为 `list[(visual_dict, heatmap_dict)]`，其中 `feats` 为 `C x T`。
- 粗检测输出生成 `video#rank` clip ID 和 segment center；细检测按 proposal center 截取 clip 特征。
- 真实模型输出已转换为 postprocess 使用的列式 prediction schema。
- CLI 顶层 `weights_mode` 会覆盖 stage 局部配置。
- latency run 会生成并由 metadata 索引：
  - `latency_results_<timestamp>.csv`
  - `latency_summary_<timestamp>.json`
  - `predictions_<timestamp>.json`
  - `meta_report_<timestamp>.json`
- duration-error 模块可读取 `eval2tower.py` 生成的 `final_result.pkl`，执行 schema 校验、独立 tIoU matching 和报告生成。
- 当前自动化测试：`29 passed`（包含 metadata、CLI 和完整 synthetic pipeline 测试）。

### 尚未完成

- 尚未在本机完成正式 raw-video benchmark；当前 worktree 没有完整 checkpoint 集合和固定视频 manifest。
- 示例配置中的 RGB/flow I3D checkpoint、`best_model_trace.pt`、coarse/fine checkpoint 和 raw-video manifest 需要替换为本机真实路径。
- cache manager 尚未接入 runner 的实际 stage 生命周期；当前中间结果保存在内存 payload 中。
- metadata 仍需补充 git commit、输入 manifest hash、PyTorch/CUDA/GPU 信息和完整命令行记录。
- 需要在真实环境核对 checkpoint 的模型架构、类别数、输入维度和 tower 名称。
- 需要用真实视频验证 flow、I3D、keypoint、heatmap 与两个 evaluator 的时序长度对齐。
- duration-error 结果必须在真实 prediction PKL 和 annotation 上运行后人工复核，不能把 fixture 结果用于论文或 rebuttal。

## 目录和入口

```text
sg-mamba/
  experiments/latency/                 latency contracts, runner, CLI
  experiments/duration_error/          duration-error analysis
  configs/latency_benchmark.example.yaml
  tests/latency/
  tests/duration_error/
  docs/superpowers/specs/               approved design
  docs/superpowers/plans/               implementation plan
```

- latency CLI：`python -m experiments.latency.cli`
- latency runner：`experiments/latency/runner.py`
- detector adapters：`experiments/latency/adapters/detectors.py`
- duration-error CLI：`python -m experiments.duration_error.cli`

## 环境

所有 Python 命令使用 `pytorch` Conda 环境：

```powershell
conda activate pytorch
cd D:\LYR\Project\Anatomically-Calibrated-Mamba-worktrees\latency-duration-experiments\sg-mamba
```

或显式指定环境：

```powershell
conda run -n pytorch python -m pytest -q
```

当前验证环境至少需要 Python、PyTorch、NumPy、PyYAML 和 pytest。真实运行还需要 OpenCV contrib、I3D、关键点推理依赖和 checkpoint。

## 测试命令

```powershell
conda run -n pytorch python -m pytest -q
conda run -n pytorch python -m pytest -q tests/latency
conda run -n pytorch python -m pytest -q tests/duration_error
```

编译关键模块：

```powershell
conda run -n pytorch python -m py_compile experiments/latency/cli.py experiments/latency/runner.py experiments/latency/adapters/detectors.py
```

## Latency 使用

### Manifest

manifest 可以是 JSON list，也可以是包含 `videos` list 的 JSON object。每个视频至少需要 `id` 和 `path`：

```json
[
  {
    "id": "example_video",
    "path": "D:/data/swallow/example_video.avi",
    "split": "test",
    "duration": 32.0
  }
]
```

### Validate

`validate` 用于 schema、adapter 顺序和测试替身验证，不是正式性能结果：

```powershell
conda run -n pytorch python -m experiments.latency.cli `
  --manifest path/to/manifest.json `
  --config configs/latency_benchmark.example.yaml `
  --mode validate `
  --weights-mode skip `
  --output-dir outputs/latency_validate
```

真实模型 factory 仍会尝试构造模型；没有 checkpoint 时应使用测试 adapter 或准备本机 checkpoint。

### Smoke

`smoke` 建议只放 1--2 个短视频，用于检查真实依赖和阶段连接：

```powershell
conda run -n pytorch python -m experiments.latency.cli `
  --manifest path/to/smoke_manifest.json `
  --config configs/latency_benchmark.example.yaml `
  --mode smoke `
  --weights-mode required `
  --output-dir outputs/latency_smoke
```

### Benchmark

正式 benchmark 前确认 checkpoint 和 manifest 存在，并固定 manifest、配置、代码 commit 和环境：

```powershell
conda run -n pytorch python -m experiments.latency.cli `
  --manifest path/to/test_manifest.json `
  --config configs/latency_benchmark.example.yaml `
  --mode benchmark `
  --weights-mode required `
  --output-dir outputs/latency_benchmark
```

运行前检查资源：

```powershell
Test-Path path/to/test_manifest.json
Test-Path pretrained/pretrained_swallow_i3d.pth
Test-Path pretrained/flow_imagenet.pt
Test-Path ckpts/best_model_trace.pt
```

### 输出解释

- `latency_results`：每个视频一行，包含各阶段耗时、总耗时、状态和失败原因。
- `latency_summary`：attempted/succeeded/failed、warm-up 视频和成功样本聚合耗时。
- `predictions`：成功视频的 postprocess prediction，不是 duration-error 报告本身。
- `meta_report`：权威 artifact index，包含文件 hash、adapter 状态、preparation 信息和运行范围。

只有 `scope=full_pipeline`、checkpoint 状态明确、失败数和输入 cohort 经人工复核的结果，才可考虑用于正式报告。

## Duration-error 使用

duration-error 独立读取 evaluator prediction PKL 和 annotation JSON：

```powershell
conda run -n pytorch python -m experiments.duration_error.cli `
  --predictions path/to/final_result.pkl `
  --annotations data/swallow/anno/swallow_singlestage.json `
  --split test `
  --label-map path/to/label_map.json `
  --tiou-thresholds 0.3 0.5 0.7 `
  --output-dir outputs/duration_error
```

该分析衡量同类别、达到指定 tIoU 阈值后的 matched boundary agreement，不替代 mAP；mAP 还反映 confidence ranking、FP/FN、分类和 localization。

## Git 整理

当前改动集中在 latency 实现、配置、测试和本文档，没有自动生成的输出文件。建议保持以下提交边界：

```text
feat: connect raw-video latency to evaluator inputs
test: cover latency artifacts and evaluator boundaries
docs: document latency and duration experiments
```

检查命令：

```powershell
git status --short --branch
git diff --check
git diff --stat
conda run -n pytorch python -m pytest -q
```

不要提交 `outputs/`、真实视频、checkpoint 或生成的 `meta_report_*.json`。完成真实 smoke/benchmark 后，记录命令、commit、配置 hash、checkpoint 标识和输出目录。
