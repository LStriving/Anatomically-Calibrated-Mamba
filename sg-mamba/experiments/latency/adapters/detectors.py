"""Detector boundaries and evaluator-compatible prediction post-processing."""
from copy import deepcopy
import gc
from pathlib import Path
import warnings
import numpy as np

from ..contracts import StructuralRunError


class CoarseDetectorAdapter:
    name = "coarse_detector"

    def __init__(self, predictor=None, feat_stride=3, num_frames=8):
        self.predictor = predictor
        self.feat_stride = feat_stride
        self.num_frames = num_frames
        self.weights_loaded = getattr(predictor, "weights_loaded", None)
        self.checkpoint_warnings = list(getattr(predictor, "checkpoint_warnings", []))

    def prepare(self, context):
        if self.predictor is None:
            raise StructuralRunError("Coarse detector factory is required; configure the stage-1 evaluator model")

    def close(self):
        return None

    def run(self, payload, context):
        coarse_input = payload.values.get("coarse_input")
        if coarse_input is None:
            coarse_input = _build_visual_input(payload, self.feat_stride, self.num_frames)
        else:
            coarse_input = _normalize_single_tower_input(coarse_input)
        _validate_single_tower_input(coarse_input, "coarse_input")
        coarse_segments = self.predictor(coarse_input, context)
        return (payload.with_value("coarse_input", coarse_input)
            .with_value("coarse_segments", coarse_segments)
            .with_value("segment_centers", _segment_centers(coarse_segments, payload)))


class FineDetectorAdapter:
    name = "fine_detector"

    def __init__(self, predictor=None, feat_stride=3, num_frames=8, heatmap_dim=576,
                 segment_duration=4.004, heatmap_size=None):
        self.predictor = predictor
        self.feat_stride = feat_stride
        self.num_frames = num_frames
        self.heatmap_dim = heatmap_dim
        self.segment_duration = segment_duration
        self.heatmap_size = heatmap_size
        self.weights_loaded = getattr(predictor, "weights_loaded", None)
        self.checkpoint_warnings = list(getattr(predictor, "checkpoint_warnings", []))

    def prepare(self, context):
        if self.predictor is None:
            raise StructuralRunError("Fine detector factory is required; configure the eval2tower two-tower model")

    def close(self):
        return None

    def run(self, payload, context):
        fine_input = payload.values.get("fine_input")
        if fine_input is None:
            fine_input = _build_two_tower_input(
                payload, self.feat_stride, self.num_frames, self.heatmap_dim,
                self.segment_duration, self.heatmap_size,
            )
        else:
            fine_input = _normalize_two_tower_input(fine_input)
        _validate_two_tower_input(fine_input)
        raw_predictions = self.predictor(fine_input, context)
        self.weights_loaded = getattr(self.predictor, "weights_loaded", self.weights_loaded)
        self.checkpoint_warnings = list(getattr(self.predictor, "checkpoint_warnings", self.checkpoint_warnings))
        return payload.with_value("fine_input", fine_input).with_value(
            "fine_predictions", _prediction_columns(raw_predictions, fine_input, payload)
        )


class PostprocessAdapter:
    """Apply eval2tower's clip-center shift and convert segment IDs to video IDs."""
    name = "postprocess"

    def __init__(self, segment_duration):
        self.segment_duration = segment_duration

    def prepare(self, context):
        return None

    def close(self):
        return None

    def run(self, payload, context):
        predictions = deepcopy(payload.require("fine_predictions"))
        centers = payload.require("segment_centers")
        required = {"seg-id", "video-id", "t-start", "t-end", "score", "label"}
        missing = required.difference(predictions)
        if missing:
            raise StructuralRunError("Fine predictions are missing fields: {}".format(", ".join(sorted(missing))))
        shift = self.segment_duration / 2
        for index, segment_id in enumerate(predictions["seg-id"]):
            if segment_id not in centers:
                raise StructuralRunError("No center recorded for segment: {}".format(segment_id))
            predictions["t-start"][index] += centers[segment_id] - shift
            predictions["t-end"][index] += centers[segment_id] - shift
        predictions["video-id"] = [video_id.split("#")[0] for video_id in predictions["video-id"]]
        return payload.with_value("predictions", predictions)


def make_coarse_detector_adapter(config, checkpoint, weights_mode="required", device=None):
    """Build the stage-1 evaluator model before the runner starts timing videos."""
    model = _make_evaluator_model(config, checkpoint, weights_mode, device)
    adapter = CoarseDetectorAdapter(
        predictor=lambda value, _: _call_model_inference(model, value),
        feat_stride=config["dataset"].get("feat_stride", 3),
        num_frames=config["dataset"].get("num_frames", 8),
    )
    _copy_weight_status(adapter, model)
    return adapter


def make_coarse_detector_from_config(config_path, checkpoint, weights_mode="required", device=None):
    """CLI-friendly stage-1 factory; config/checkpoint work happens before timing."""
    from libs.core import load_config

    return make_coarse_detector_adapter(
        load_config(config_path), checkpoint, weights_mode=weights_mode, device=device,
    )


def make_two_tower_detector_adapter(config, config2, checkpoint, tower_name,
                                    weights_mode="required", device=None):
    """Build eval2tower's model and retain its exact list-of-pairs input contract."""
    from libs.modeling import make_meta_arch, make_two_tower

    visual = make_meta_arch(config["model_name"], **config["model"])
    heatmap = make_meta_arch(config2["model_name"], **config2["model"])
    model = make_two_tower(tower_name, visual, heatmap, config, config2, **config["two_tower"])
    model = _prepare_model(model, checkpoint, weights_mode, device, config.get("devices"))
    adapter = FineDetectorAdapter(
        predictor=lambda value, _: _call_model_inference(model, value),
        feat_stride=config["dataset"].get("feat_stride", 3),
        num_frames=config["dataset"].get("num_frames", 8),
        heatmap_dim=config2["dataset"].get("input_dim", 576),
        segment_duration=config.get("seg_duration", 4.004),
        heatmap_size=_heatmap_size_from_config(config2),
    )
    _copy_weight_status(adapter, model)
    return adapter


def make_two_tower_detector_from_config(config_path, config2_path, checkpoint=None, tower_name="LogitsAvg",
                                        weights_mode="required", device=None, actions=None):
    """CLI-friendly eval2tower factory using the evaluator's two YAML configs."""
    if actions is not None:
        return make_action_two_tower_detector_from_config(
            config_path=config_path,
            config2_path=config2_path,
            actions=actions,
            tower_name=tower_name,
            weights_mode=weights_mode,
            device=device,
        )
    from libs.core import load_config

    return make_two_tower_detector_adapter(
        load_config(config_path), load_config(config2_path), checkpoint, tower_name,
        weights_mode=weights_mode, device=device,
    )


def make_action_two_tower_detector_adapter(actions, predictor_builder=None,
                                           feat_stride=3, num_frames=8,
                                           heatmap_dim=576, segment_duration=4.004,
                                           heatmap_size=None):
    """Build a fine detector from the seven action-specific stage-2 models."""
    if not isinstance(actions, (list, tuple)) or len(actions) != 7:
        raise StructuralRunError("Action two-tower fine detector requires exactly 7 action models")
    action_specs = []
    for action in actions:
        if not isinstance(action, dict) or "name" not in action or "checkpoint" not in action:
            raise StructuralRunError("Each action model requires name and checkpoint")
        action_specs.append(dict(action))
    adapter = FineDetectorAdapter(
        predictor=_ActionTwoTowerEnsemble(
            action_specs, predictor_builder or _default_action_predictor_builder,
        ),
        feat_stride=feat_stride,
        num_frames=num_frames,
        heatmap_dim=heatmap_dim,
        segment_duration=segment_duration,
        heatmap_size=heatmap_size,
    )
    return adapter


def make_action_two_tower_detector_from_config(config_path, config2_path, actions,
                                               tower_name="LogitsAvg",
                                               weights_mode="required", device=None):
    """CLI-friendly seven-model fine detector mirroring eval2stage action ckpts."""
    from libs.core import load_config

    config = load_config(config_path)
    config2 = load_config(config2_path)

    def builder(_action_name, checkpoint):
        adapter = make_two_tower_detector_adapter(
            config, config2, checkpoint, tower_name,
            weights_mode=weights_mode, device=device,
        )
        return _SingleActionPredictor(adapter)

    return make_action_two_tower_detector_adapter(
        actions=actions,
        predictor_builder=builder,
        feat_stride=config["dataset"].get("feat_stride", 3),
        num_frames=config["dataset"].get("num_frames", 8),
        heatmap_dim=config2["dataset"].get("input_dim", 576),
        segment_duration=config.get("seg_duration", 4.004),
        heatmap_size=_heatmap_size_from_config(config2),
    )


def _make_evaluator_model(config, checkpoint, weights_mode, device):
    from libs.modeling import make_meta_arch

    model = make_meta_arch(config["model_name"], **config["model"])
    return _prepare_model(model, checkpoint, weights_mode, device, config.get("devices"))


class _ActionTwoTowerEnsemble:
    def __init__(self, actions, predictor_builder):
        self.actions = actions
        self.predictor_builder = predictor_builder
        self.weights_loaded = None
        self.checkpoint_warnings = []

    def __call__(self, batch, context):
        results = []
        loaded = []
        warnings_seen = []
        for action in self.actions:
            predictor = self.predictor_builder(action["name"], action["checkpoint"])
            try:
                action_results = predictor(batch, context)
                if isinstance(action_results, dict):
                    action_results = [action_results]
                if not isinstance(action_results, list):
                    raise StructuralRunError("Action model {} returned non-list results".format(action["name"]))
                results.extend(action_results)
                action_loaded = getattr(predictor, "weights_loaded", None)
                if action_loaded is not None:
                    loaded.append(bool(action_loaded))
                warnings_seen.extend(getattr(predictor, "checkpoint_warnings", []))
            finally:
                _release_predictor(predictor)
        self.weights_loaded = all(loaded) if loaded else None
        self.checkpoint_warnings = warnings_seen
        return results


class _SingleActionPredictor:
    def __init__(self, adapter):
        self.adapter = adapter
        self.weights_loaded = getattr(adapter, "weights_loaded", None)
        self.checkpoint_warnings = list(getattr(adapter, "checkpoint_warnings", []))

    def __call__(self, batch, context):
        return self.adapter.predictor(batch, context)

    def close(self):
        self.adapter.predictor = None
        self.adapter = None


def _default_action_predictor_builder(action_name, checkpoint):
    raise StructuralRunError(
        "Action model {} requires make_action_two_tower_detector_from_config or predictor_builder".format(action_name)
    )


def _release_predictor(predictor):
    close = getattr(predictor, "close", None)
    if callable(close):
        close()
    del predictor
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except (ImportError, RuntimeError):
        return None


def _prepare_model(model, checkpoint, weights_mode, device, devices=None):
    import torch

    target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(target).eval()
    if target.type == "cuda" and devices:
        model = torch.nn.DataParallel(model, device_ids=list(devices))
        model.eval()
    model.weights_loaded = False
    model.checkpoint_warnings = []
    if checkpoint is None or not Path(checkpoint).is_file():
        message = "detector checkpoint unavailable: {}".format(checkpoint)
        if weights_mode == "required":
            raise StructuralRunError(message)
        model.checkpoint_warnings.append(message)
        warnings.warn(message, RuntimeWarning)
        return model
    try:
        state = torch.load(checkpoint, map_location=target)
        state = state.get("state_dict_ema", state.get("state_dict", state))
        model.load_state_dict(state)
        model.weights_loaded = True
    except (OSError, RuntimeError, KeyError, TypeError) as error:
        if weights_mode == "required":
            raise StructuralRunError("unable to load detector checkpoint: {}".format(checkpoint)) from error
        message = "detector random weights used after checkpoint load failure: {}".format(error)
        model.checkpoint_warnings.append(message)
        warnings.warn(message, RuntimeWarning)
    return model


def _call_model_inference(model, value):
    import torch

    with torch.inference_mode():
        return model(value)


def _copy_weight_status(adapter, model):
    adapter.weights_loaded = getattr(model, "weights_loaded", None)
    adapter.checkpoint_warnings = list(getattr(model, "checkpoint_warnings", []))


def _validate_single_tower_input(value, name):
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict):
        raise StructuralRunError("{} must be a one-item evaluator batch of dicts".format(name))
    _validate_evaluator_item(value[0], name + "[0]")


def _normalize_single_tower_input(value):
    return [_normalize_evaluator_item(item) for item in value]


def _normalize_two_tower_input(value):
    return [
        (_normalize_evaluator_item(pair[0]), _normalize_evaluator_item(pair[1]))
        for pair in value
    ]


def _validate_two_tower_input(value):
    if not isinstance(value, list) or not value:
        raise StructuralRunError("fine_input must be a non-empty MultiModalDataset batch")
    for index, pair in enumerate(value):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise StructuralRunError("fine_input[{}] must be (visual_dict, heatmap_dict)".format(index))
        _validate_evaluator_item(pair[0], "fine_input[{}][0]".format(index))
        _validate_evaluator_item(pair[1], "fine_input[{}][1]".format(index))


def _build_visual_input(payload, feat_stride, num_frames):
    rgb = np.asarray(payload.require("rgb_i3d_features"), dtype=np.float32)
    flow = np.asarray(payload.require("flow_i3d_features"), dtype=np.float32)
    if rgb.ndim != 2 or flow.ndim != 2 or rgb.shape[0] != flow.shape[0]:
        raise StructuralRunError("I3D features must be matching (time, channels) arrays")
    features = np.concatenate((rgb, flow), axis=1)
    return [_evaluator_item(payload, features, feat_stride, num_frames)]


def _build_two_tower_input(payload, feat_stride, num_frames, heatmap_dim, segment_duration, heatmap_size=None):
    visual = _build_visual_input(payload, feat_stride, num_frames)[0]
    heatmap = np.asarray(payload.require("skeleton_features"), dtype=np.float32)
    if heatmap.ndim != 3:
        raise StructuralRunError("skeleton_features must have shape (time, height, width)")
    heatmap = _resize_heatmap_frames(heatmap, heatmap_size or _square_size(heatmap_dim))
    centers = payload.values.get("segment_centers") or {
        payload.require("video")["id"] + "#0": 0.0
    }
    pairs = []
    for segment_id, center in centers.items():
        visual_clip = _clip_features(_features_to_numpy(visual["feats"]).T, center, payload, feat_stride, segment_duration)
        heatmap_clip = _clip_features(heatmap, center, payload, feat_stride, segment_duration)
        pairs.append((
            _evaluator_item(payload, visual_clip, feat_stride, num_frames, video_id=segment_id,
                            duration=segment_duration),
            _heatmap_evaluator_item(payload, heatmap_clip, feat_stride, num_frames, video_id=segment_id,
                                    duration=segment_duration),
        ))
    return pairs


def _evaluator_item(payload, time_features, feat_stride, num_frames, video_id=None, duration=None):
    video = payload.require("video")
    features = np.asarray(time_features, dtype=np.float32)
    fps = float(payload.values.get("fps", video.get("fps", 1.0)))
    duration = float(duration if duration is not None else video.get(
        "duration", len(features) * feat_stride / max(fps, 1e-6)
    ))
    return {
        "video_id": video_id or video["id"],
        "feats": _as_feature_tensor(features.T.copy()),
        "segments": None,
        "labels": None,
        "fps": fps,
        "duration": duration,
        "feat_stride": feat_stride,
        "feat_num_frames": num_frames,
    }


def _heatmap_evaluator_item(payload, time_features, feat_stride, num_frames, video_id=None, duration=None):
    item = _evaluator_item(payload, np.zeros((len(time_features), 1), dtype=np.float32),
                           feat_stride, num_frames, video_id=video_id, duration=duration)
    item["feats"] = _as_feature_tensor(np.asarray(time_features, dtype=np.float32)[None, ...])
    return item


def _clip_features(features, center, payload, feat_stride, segment_duration):
    features = np.asarray(features, dtype=np.float32)
    fps = float(payload.values.get("fps", payload.require("video").get("fps", 1.0)))
    clip_length = max(1, int(round(segment_duration * fps / feat_stride)))
    center_index = int(round(float(center) * fps / feat_stride))
    start = center_index - clip_length // 2
    stop = start + clip_length
    left_pad = max(0, -start)
    right_pad = max(0, stop - len(features))
    clipped = features[max(0, start):min(len(features), stop)]
    if left_pad or right_pad:
        clipped = np.pad(clipped, [(left_pad, right_pad)] + [(0, 0)] * (features.ndim - 1), mode="edge")
    return clipped


def _square_size(output_dim):
    side = int(round(output_dim ** 0.5))
    if side * side != output_dim:
        raise StructuralRunError("heatmap input_dim must be a square feature size")
    return side


def _resize_heatmap_frames(features, side):
    import torch
    import torch.nn.functional as functional

    values = torch.from_numpy(features).unsqueeze(1)
    return functional.interpolate(values, size=(side, side), mode="bilinear", align_corners=False)[:, 0].numpy()


def _heatmap_size_from_config(config):
    dataset = config.get("dataset", {})
    video_stem = config.get("video_stem", {})
    return dataset.get("resize_to") or video_stem.get("image_size")


def _segment_centers(results, payload):
    centers = {}
    if isinstance(results, list):
        for result_index, result in enumerate(results):
            segments = result.get("segments") if isinstance(result, dict) else None
            video_id = result.get("video_id") if isinstance(result, dict) else None
            if segments is None or video_id is None:
                continue
            if hasattr(segments, "detach"):
                segments = segments.detach().cpu().numpy()
            for segment_index, segment in enumerate(np.asarray(segments)):
                segment_id = "{}#{}".format(video_id, segment_index)
                centers[segment_id] = float((segment[0] + segment[1]) / 2)
    if not centers:
        video = payload.values.get("video")
        if video is not None:
            centers[video["id"] + "#0"] = float(video.get("duration", 0.0)) / 2
    return centers


def _prediction_columns(results, fine_input, payload):
    if isinstance(results, dict):
        return results
    if not isinstance(results, list):
        raise StructuralRunError("Fine detector must return a list of result dicts")
    columns = {"seg-id": [], "video-id": [], "t-start": [], "t-end": [], "score": [], "label": []}
    for result_index, result in enumerate(results):
        if not isinstance(result, dict):
            raise StructuralRunError("Fine detector result {} must be a dict".format(result_index))
        if "segments" not in result:
            continue
        segments = result["segments"]
        if hasattr(segments, "detach"):
            segments = segments.detach().cpu().numpy()
        scores = result.get("scores", [])
        labels = result.get("labels", [])
        if hasattr(scores, "detach"):
            scores = scores.detach().cpu().numpy()
        if hasattr(labels, "detach"):
            labels = labels.detach().cpu().numpy()
        video_id = result.get("video_id", payload.require("video")["id"])
        segments = np.asarray(segments)
        for segment_index, segment in enumerate(segments):
            segment_id = _result_segment_id(video_id, segment_index, len(segments))
            columns["seg-id"].append(segment_id)
            columns["video-id"].append(segment_id)
            columns["t-start"].append(float(segment[0]))
            columns["t-end"].append(float(segment[1]))
            columns["score"].append(float(np.asarray(scores)[segment_index]))
            columns["label"].append(int(np.asarray(labels)[segment_index]))
    return columns


def _result_segment_id(video_id, segment_index, count):
    if "#" in video_id and count == 1:
        return video_id
    return "{}#{}".format(video_id, segment_index)


def _validate_evaluator_item(value, name):
    if not isinstance(value, dict):
        raise StructuralRunError("{} must be an evaluator item dict".format(name))
    required = {"video_id", "feats", "feat_stride", "feat_num_frames"}
    missing = required.difference(value)
    if missing:
        raise StructuralRunError("{} is missing fields: {}".format(name, ", ".join(sorted(missing))))
    try:
        import torch
    except ImportError as error:
        raise StructuralRunError("PyTorch is required for evaluator feature tensors") from error
    if not isinstance(value["feats"], torch.Tensor):
        raise StructuralRunError("{} feats must be a torch.Tensor".format(name))


def _normalize_evaluator_item(value):
    normalized = dict(value)
    normalized["feats"] = _as_feature_tensor(normalized["feats"])
    return normalized


def _as_feature_tensor(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.float().contiguous()
    return torch.from_numpy(np.ascontiguousarray(np.asarray(value, dtype=np.float32)))


def _features_to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)
