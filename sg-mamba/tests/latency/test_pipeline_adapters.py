"""Contract tests for raw-video latency pipeline adapters."""
import numpy as np
import torch
import pytest

from experiments.latency.contracts import Payload, StructuralRunError


class _FeatureModel:
    def extract_features(self, batch):
        return batch.mean(dim=(2, 3, 4), keepdim=True)


def test_i3d_adapter_emits_rgb_and_flow_features_from_eight_frame_windows():
    """Dropping RGB/flow normalization or channel order changes detector inputs."""
    from experiments.latency.adapters.i3d import I3DExtractAdapter

    frames = [np.full((4, 4, 3), 127, dtype=np.uint8) for _ in range(8)]
    flow = np.zeros((8, 4, 4, 2), dtype=np.float32)
    adapter = I3DExtractAdapter(rgb_model=_FeatureModel(), flow_model=_FeatureModel(), image_size=4)
    adapter.prepare({})
    result = adapter.run(Payload({"frames": frames, "flow": flow}), {})

    assert result.require("rgb_i3d_features").shape == (1, 3)
    assert result.require("flow_i3d_features").shape == (1, 2)
    assert np.allclose(result.require("rgb_i3d_features"), -1 / 255, atol=1e-6)


def test_i3d_factory_rejects_missing_required_checkpoints(tmp_path):
    """Treating absent required weights as random weights would invalidate a benchmark run."""
    from experiments.latency.adapters.i3d import make_i3d_adapter

    with pytest.raises(StructuralRunError, match="checkpoint does not exist"):
        make_i3d_adapter(
            rgb_checkpoint=tmp_path / "missing-rgb.pth",
            flow_checkpoint=tmp_path / "missing-flow.pth",
            weights_mode="required",
        )


class _KeypointProcessor:
    def infer_keypoints(self, frames, kalman=False, normal_kalman=False):
        assert kalman is False
        return np.full((len(frames), 8, 2), 0.25, dtype=np.float32), np.ones((len(frames), 8)), 4, 4


def test_keypoint_and_kalman_adapters_preserve_reference_data_boundaries():
    """Combining extraction with smoothing would hide the separately timed Kalman stage."""
    from experiments.latency.adapters.keypoints import KalmanSmoothAdapter, KeypointExtractAdapter

    raw = np.full((2, 8, 2), 0.25, dtype=np.float32)
    payload = Payload({"frames": [np.zeros((4, 4, 3), dtype=np.uint8)] * 2})
    keypoints = KeypointExtractAdapter(processor=_KeypointProcessor()).run(payload, {})
    smoothed = KalmanSmoothAdapter(smoother=lambda points, confidence: points + 0.5).run(keypoints, {})

    assert np.array_equal(keypoints.require("keypoints"), raw)
    assert np.array_equal(smoothed.require("smoothed_keypoints"), raw + 0.5)


def test_postprocess_adapter_uses_eval2tower_shift_result_without_mutating_input():
    """Removing segment-center shifting would report clip-relative rather than video-relative boundaries."""
    from experiments.latency.adapters.detectors import PostprocessAdapter

    predictions = {
        "seg-id": ["video#0"], "video-id": ["video#0"],
        "t-start": np.array([0.5]), "t-end": np.array([1.5]),
        "score": np.array([0.9]), "label": np.array([1]),
    }
    result = PostprocessAdapter(segment_duration=4.0).run(
        Payload({"fine_predictions": predictions, "segment_centers": {"video#0": 7.0}}), {}
    ).require("predictions")

    assert result["video-id"] == ["video"]
    assert result["t-start"].tolist() == [5.5]
    assert predictions["t-start"].tolist() == [0.5]


def test_skeleton_and_detector_adapters_keep_model_outputs_in_named_payload_fields():
    """Changing payload names would disconnect the independently timed downstream stages."""
    from experiments.latency.adapters.detectors import CoarseDetectorAdapter, FineDetectorAdapter
    from experiments.latency.adapters.keypoints import SkeletonEncodeAdapter

    payload = Payload({"smoothed_keypoints": np.zeros((2, 8, 2)), "keypoint_confidences": np.ones((2, 8))})
    skeleton = SkeletonEncodeAdapter(encoder=lambda points, confidence: np.ones((2, 4, 4))).run(payload, {})
    coarse = CoarseDetectorAdapter(predictor=lambda value, context: [{"center": 2.0}]).run(skeleton, {})
    fine = FineDetectorAdapter(predictor=lambda value, context: {"score": [0.8]}).run(coarse, {})

    assert skeleton.require("skeleton_features").shape == (2, 4, 4)
    assert coarse.require("coarse_segments") == [{"center": 2.0}]
    assert fine.require("fine_predictions") == {"score": [0.8]}
