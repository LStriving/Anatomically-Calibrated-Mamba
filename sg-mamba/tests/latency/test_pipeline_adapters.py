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

    evaluator_item = {
        "video_id": "video#0", "feats": np.zeros((4, 2)),
        "feat_stride": 1, "feat_num_frames": 1,
    }
    payload = Payload({"smoothed_keypoints": np.zeros((2, 8, 2)), "keypoint_confidences": np.ones((2, 8)),
                       "coarse_input": [evaluator_item],
                       "fine_input": [([evaluator_item][0], dict(evaluator_item))]})
    skeleton = SkeletonEncodeAdapter(encoder=lambda points, confidence: np.ones((2, 4, 4))).run(payload, {})
    coarse = CoarseDetectorAdapter(predictor=lambda value, context: [{"center": 2.0}]).run(skeleton, {})
    fine = FineDetectorAdapter(predictor=lambda value, context: {"score": [0.8]}).run(coarse, {})

    assert skeleton.require("skeleton_features").shape == (2, 4, 4)
    assert coarse.require("coarse_segments") == [{"center": 2.0}]
    assert fine.require("fine_predictions") == {"score": [0.8]}


def test_detectors_build_real_evaluator_inputs_from_feature_payload():
    """Raw-video stages must connect to the evaluator without temporary feature files."""
    from experiments.latency.adapters.detectors import CoarseDetectorAdapter, FineDetectorAdapter

    payload = Payload({
        "video": {"id": "video", "path": "video.avi", "duration": 2.0},
        "fps": 10.0,
        "rgb_i3d_features": np.zeros((4, 2), dtype=np.float32),
        "flow_i3d_features": np.ones((4, 2), dtype=np.float32),
        "skeleton_features": np.ones((4, 4, 4), dtype=np.float32),
    })
    seen = {}

    def coarse_predictor(batch, _):
        seen["coarse"] = batch
        return []

    def fine_predictor(batch, _):
        seen["fine"] = batch
        return []

    payload = CoarseDetectorAdapter(coarse_predictor, feat_stride=3, num_frames=8).run(payload, {})
    payload = FineDetectorAdapter(fine_predictor, feat_stride=3, num_frames=8, heatmap_dim=4).run(payload, {})

    assert isinstance(seen["coarse"][0]["feats"], torch.Tensor)
    assert isinstance(seen["fine"][0][0]["feats"], torch.Tensor)
    assert isinstance(seen["fine"][0][1]["feats"], torch.Tensor)
    assert seen["coarse"][0]["feats"].shape == (4, 4)
    assert seen["fine"][0][0]["feats"].shape[0] == 4
    assert seen["fine"][0][1]["feats"].ndim == 4
    assert seen["fine"][0][0]["video_id"] == "video#0"


def test_fine_detector_keeps_heatmap_branch_as_video_tensor():
    """The heatmap VideoMamba tower expects C x T x H x W, not flattened C x T features."""
    from experiments.latency.adapters.detectors import FineDetectorAdapter

    payload = Payload({
        "video": {"id": "video", "path": "video.avi", "duration": 2.0},
        "fps": 8.0,
        "rgb_i3d_features": np.zeros((4, 2), dtype=np.float32),
        "flow_i3d_features": np.ones((4, 2), dtype=np.float32),
        "skeleton_features": np.ones((4, 12, 12), dtype=np.float32),
        "segment_centers": {"video#0": 0.5},
    })
    seen = {}

    def fine_predictor(batch, _):
        seen["batch"] = batch
        return []

    FineDetectorAdapter(
        fine_predictor, feat_stride=1, num_frames=2, heatmap_size=8,
        segment_duration=1.0,
    ).run(payload, {})

    heatmap_feats = seen["batch"][0][1]["feats"]
    assert isinstance(heatmap_feats, torch.Tensor)
    assert heatmap_feats.shape == (1, 8, 8, 8)


def test_detectors_normalize_injected_evaluator_inputs_to_tensors():
    """The evaluator calls torch padding on feats, so injected inputs cannot remain numpy arrays."""
    from experiments.latency.adapters.detectors import CoarseDetectorAdapter, FineDetectorAdapter

    coarse_seen = {}
    fine_seen = {}
    item = {"video_id": "video#0", "feats": np.zeros((2, 4), dtype=np.float32),
            "feat_stride": 3, "feat_num_frames": 8}
    payload = Payload({
        "video": {"id": "video", "path": "video.avi", "duration": 2.0},
        "coarse_input": [dict(item)],
        "fine_input": [(dict(item), dict(item))],
    })

    def coarse_predictor(batch, _):
        coarse_seen["batch"] = batch
        return []

    def fine_predictor(batch, _):
        fine_seen["batch"] = batch
        return []

    payload = CoarseDetectorAdapter(predictor=coarse_predictor).run(payload, {})
    FineDetectorAdapter(predictor=fine_predictor).run(payload, {})

    assert isinstance(coarse_seen["batch"][0]["feats"], torch.Tensor)
    assert isinstance(fine_seen["batch"][0][0]["feats"], torch.Tensor)
    assert isinstance(fine_seen["batch"][0][1]["feats"], torch.Tensor)


def test_fine_detector_normalizes_model_results_for_postprocess():
    from experiments.latency.adapters.detectors import FineDetectorAdapter

    payload = Payload({
        "video": {"id": "video", "path": "video.avi", "duration": 2.0},
        "fine_input": [({"video_id": "video#0", "feats": np.zeros((2, 4)),
                         "feat_stride": 3, "feat_num_frames": 8},
                        {"video_id": "video#0", "feats": np.zeros((2, 4)),
                         "feat_stride": 3, "feat_num_frames": 8})],
    })
    result = FineDetectorAdapter(
        predictor=lambda batch, _: [{"video_id": "video#0", "segments": np.array([[1.0, 2.0]]),
                                     "scores": np.array([0.8]), "labels": np.array([3])}]
    ).run(payload, {}).require("fine_predictions")

    assert result["seg-id"] == ["video#0"]
    assert result["t-start"] == [1.0]
    assert result["label"] == [3]


def test_action_two_tower_ensemble_builds_one_model_per_action_and_merges_results():
    """Using one stage-2 checkpoint for all classes would miss eval2stage's per-action model contract."""
    from experiments.latency.adapters.detectors import make_action_two_tower_detector_adapter

    built = []

    def builder(action_name, checkpoint):
        built.append((action_name, checkpoint))

        def predictor(batch, _):
            return [{
                "video_id": batch[0][0]["video_id"],
                "segments": np.asarray([[0.1, 0.2]], dtype=np.float32),
                "scores": np.asarray([0.9], dtype=np.float32),
                "labels": np.asarray([len(built)], dtype=np.int64),
            }]

        return predictor

    adapter = make_action_two_tower_detector_adapter(
        actions=[
            {"name": "oral", "checkpoint": "oral.pth.tar"},
            {"name": "soft_palate", "checkpoint": "soft.pth.tar"},
            {"name": "hyoid", "checkpoint": "hyoid.pth.tar"},
            {"name": "larynx", "checkpoint": "larynx.pth.tar"},
            {"name": "epiglottis", "checkpoint": "epi.pth.tar"},
            {"name": "ues", "checkpoint": "ues.pth.tar"},
            {"name": "bolus", "checkpoint": "bolus.pth.tar"},
        ],
        predictor_builder=builder,
    )
    payload = Payload({
        "video": {"id": "video", "path": "video.avi", "duration": 2.0, "fps": 30},
        "fine_input": [({"video_id": "video#0", "feats": np.zeros((2, 4), dtype=np.float32),
                         "feat_stride": 3, "feat_num_frames": 8},
                        {"video_id": "video#0", "feats": np.zeros((2, 4), dtype=np.float32),
                         "feat_stride": 3, "feat_num_frames": 8})],
    })

    result = adapter.run(payload, {}).require("fine_predictions")

    assert len(built) == 7
    assert built[0] == ("oral", "oral.pth.tar")
    assert len(result["label"]) == 7


def test_legacy_two_tower_factory_delegates_actions_to_action_ensemble(monkeypatch):
    """Older configs may keep the single-model factory path while adding action checkpoints."""
    from experiments.latency.adapters import detectors

    seen = {}

    def fake_action_factory(**kwargs):
        seen.update(kwargs)
        return "adapter"

    monkeypatch.setattr(detectors, "make_action_two_tower_detector_from_config", fake_action_factory)
    actions = [{"name": str(index), "checkpoint": str(index)} for index in range(7)]

    result = detectors.make_two_tower_detector_from_config(
        config_path="visual.yaml",
        config2_path="heatmap.yaml",
        checkpoint=None,
        tower_name="LogitsAvg",
        weights_mode="skip",
        actions=actions,
    )

    assert result == "adapter"
    assert seen["actions"] == actions
    assert seen["config_path"] == "visual.yaml"
    assert seen["config2_path"] == "heatmap.yaml"
    assert seen["tower_name"] == "LogitsAvg"
    assert seen["weights_mode"] == "skip"


def test_reference_skeleton_adapter_preserves_heatmap_branch_geometry():
    """Replacing the reference heatmap encoding would feed a different stage-2 representation."""
    from experiments.latency.adapters.keypoints import make_skeleton_adapter

    adapter = make_skeleton_adapter(sigma=0.6, crop_mode="none")
    payload = Payload({
        "smoothed_keypoints": np.full((2, 8, 2), 0.5, dtype=np.float32),
        "keypoint_confidences": np.ones((2, 8), dtype=np.float32),
        "frame_height": 12,
        "frame_width": 20,
    })

    features = adapter.run(payload, {}).require("skeleton_features")

    assert features.shape == (2, 12, 20)
    assert features.dtype == np.float32
    assert float(features.max()) == pytest.approx(1.0)


def test_keypoint_factory_reports_a_missing_traced_checkpoint_before_loading_models(tmp_path):
    """A missing landmark model must stop the run before it can emit invalid keypoints."""
    from experiments.latency.adapters.keypoints import make_keypoint_adapter

    with pytest.raises(StructuralRunError, match="checkpoint does not exist"):
        make_keypoint_adapter(model_path=tmp_path / "missing-keypoints.pt")
