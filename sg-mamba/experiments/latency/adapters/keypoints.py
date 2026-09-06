"""Frame-array keypoint and Kalman stages using the project's NPY API."""
from pathlib import Path
import numpy as np

from ..contracts import StructuralRunError


class KeypointExtractAdapter:
    name = "keypoint_extract"

    def __init__(self, processor=None):
        self.processor = processor

    def prepare(self, context):
        if self.processor is None:
            raise StructuralRunError(
                "Keypoint processor factory is required; use libs.utils.inference_keypoints_npy_api.VideoKeypointProcessor"
            )

    def close(self):
        return None

    def run(self, payload, context):
        keypoints, confidences, height, width = self.processor.infer_keypoints(
            np.asarray(payload.require("frames")), kalman=False, normal_kalman=False
        )
        return (payload.with_value("keypoints", np.asarray(keypoints))
                .with_value("keypoint_confidences", np.asarray(confidences))
                .with_value("frame_height", height).with_value("frame_width", width))


def make_keypoint_adapter(model_path, image_width=192, image_height=256, batch_size=32,
                          num_workers=4, sigma=0.6, crop_mode="auto"):
    """Construct the traced landmark processor outside the measured video path."""
    path = Path(model_path)
    if not path.is_file():
        raise StructuralRunError("Keypoint checkpoint does not exist: {}".format(path))
    try:
        from libs.utils.inference_keypoints_npy_api import VideoKeypointProcessor
    except ImportError as error:
        raise StructuralRunError(
            "Keypoint dependencies are unavailable; install pykalman and scikit-video for inference_keypoints_npy_api"
        ) from error
    try:
        processor = VideoKeypointProcessor(
            str(path), image_width=image_width, image_height=image_height,
            batch_size=batch_size, num_workers=num_workers, sigma=sigma, crop_mode=crop_mode,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise StructuralRunError("Unable to load keypoint checkpoint: {}".format(path)) from error
    adapter = KeypointExtractAdapter(processor=processor)
    adapter.weights_loaded = True
    adapter.checkpoint_warnings = []
    return adapter


class KalmanSmoothAdapter:
    name = "kalman_smooth"

    def __init__(self, smoother=None):
        self.smoother = smoother

    def prepare(self, context):
        if self.smoother is None:
            self.smoother = _reference_smoother

    def close(self):
        return None

    def run(self, payload, context):
        return payload.with_value(
            "smoothed_keypoints", self.smoother(payload.require("keypoints"), payload.require("keypoint_confidences"))
        )


class SkeletonEncodeAdapter:
    """Turn smoothed landmark tracks into the skeleton representation used by fine detection."""
    name = "skeleton_encode"

    def __init__(self, encoder=None):
        self.encoder = encoder

    def prepare(self, context):
        if self.encoder is None:
            raise StructuralRunError(
                "Skeleton encoder factory is required; use inference_keypoints_npy_api heatmap generation"
            )

    def close(self):
        return None

    def run(self, payload, context):
        keypoints = payload.require("smoothed_keypoints")
        confidences = payload.require("keypoint_confidences")
        if hasattr(self.encoder, "encode"):
            features = self.encoder.encode(
                keypoints,
                confidences,
                payload.require("frame_height"),
                payload.require("frame_width"),
            )
        else:
            features = self.encoder(keypoints, confidences)
        return payload.with_value(
            "skeleton_features",
            features,
        )


def make_skeleton_adapter(sigma=0.6, crop_mode="auto"):
    """Build the heatmap branch used by ``VideoKeypointProcessor.infer_heatmaps``.

    This stage deliberately accepts already-smoothed landmarks so the Kalman
    work stays in its own timed stage rather than being hidden in heatmap
    extraction.
    """
    return SkeletonEncodeAdapter(encoder=_ReferenceSkeletonEncoder(sigma=sigma, crop_mode=crop_mode))


class _ReferenceSkeletonEncoder:
    """Numpy implementation of the heatmap portion of the NPY reference API."""

    _SKELETON = np.array([(0, 1), (0, 2), (1, 2), (0, 4), (3, 4), (3, 6), (5, 6), (5, 7), (6, 7)])

    def __init__(self, sigma, crop_mode):
        if crop_mode not in {"auto", "none"}:
            raise StructuralRunError("Skeleton crop_mode must be 'auto' or 'none'")
        self.sigma = sigma
        self.crop_mode = crop_mode

    def encode(self, keypoints, confidences, height, width):
        keypoints = np.asarray(keypoints, dtype=np.float32)
        confidences = np.asarray(confidences, dtype=np.float32)
        if keypoints.ndim != 3 or keypoints.shape[1:] != (8, 2):
            raise StructuralRunError("Smoothed keypoints must have shape (frames, 8, 2)")
        if confidences.shape != keypoints.shape[:2]:
            raise StructuralRunError("Keypoint confidences must have shape (frames, 8)")
        if not isinstance(height, (int, np.integer)) or not isinstance(width, (int, np.integer)) or height <= 0 or width <= 0:
            raise StructuralRunError("Frame height and width must be positive integers")
        points = keypoints.copy()
        points[:, :, 0] *= width
        points[:, :, 1] *= height
        keypoint_maps = np.zeros((len(points), height, width), dtype=np.float32)
        edge_maps = np.zeros_like(keypoint_maps)
        for index, (frame_points, frame_confidence) in enumerate(zip(points, confidences)):
            _draw_points(keypoint_maps[index], frame_points, frame_confidence, self.sigma)
            links = self._SKELETON
            _draw_limbs(
                edge_maps[index], frame_points[links[:, 0]], frame_points[links[:, 1]],
                frame_confidence[links[:, 0]], frame_confidence[links[:, 1]], self.sigma,
            )
        if self.crop_mode == "auto":
            nonzero = np.nonzero(keypoint_maps)
            if len(nonzero[0]):
                keypoint_maps = keypoint_maps[:, nonzero[1].min():nonzero[1].max() + 1, nonzero[2].min():nonzero[2].max() + 1]
                edge_maps = edge_maps[:, nonzero[1].min():nonzero[1].max() + 1, nonzero[2].min():nonzero[2].max() + 1]
        return (keypoint_maps + edge_maps) * 0.5


def _draw_points(canvas, centers, values, sigma):
    for center, value in zip(centers, values):
        _draw_gaussian_or_limb(canvas, center, center, value, value, sigma)


def _draw_limbs(canvas, starts, ends, start_values, end_values, sigma):
    for start, end, start_value, end_value in zip(starts, ends, start_values, end_values):
        _draw_gaussian_or_limb(canvas, start, end, start_value, end_value, sigma)


def _draw_gaussian_or_limb(canvas, start, end, start_value, end_value, sigma):
    height, width = canvas.shape
    min_x, max_x = max(int(min(start[0], end[0]) - 3 * sigma), 0), min(int(max(start[0], end[0]) + 3 * sigma) + 1, width)
    min_y, max_y = max(int(min(start[1], end[1]) - 3 * sigma), 0), min(int(max(start[1], end[1]) + 3 * sigma) + 1, height)
    x, y = np.arange(min_x, max_x, dtype=np.float32), np.arange(min_y, max_y, dtype=np.float32)
    if not len(x) or not len(y):
        return
    xx, yy = np.meshgrid(x, y)
    length_squared = float(np.sum((start - end) ** 2))
    if length_squared < 1:
        distance_squared = (xx - start[0]) ** 2 + (yy - start[1]) ** 2
    else:
        projection = ((xx - start[0]) * (end[0] - start[0]) + (yy - start[1]) * (end[1] - start[1])) / length_squared
        projection = np.clip(projection, 0, 1)
        distance_squared = (xx - (start[0] + projection * (end[0] - start[0]))) ** 2 + (yy - (start[1] + projection * (end[1] - start[1]))) ** 2
    patch = np.exp(-distance_squared / (2 * sigma ** 2)) * min(1, start_value, end_value)
    canvas[min_y:max_y, min_x:max_x] = np.maximum(canvas[min_y:max_y, min_x:max_x], patch)


def _reference_smoother(keypoints, confidences):
    """Match the two-pass confidence-aware fusion in the NPY reference API."""
    try:
        from libs.utils.inference_keypoints_npy_api import (
            kalman_filter_with_confidence, kalman_filter_without_confidence, mixed_keypoints_weighted,
        )
    except ImportError as error:
        raise StructuralRunError("pykalman is required for the reference keypoint smoother") from error
    if len(keypoints) <= 1:
        return keypoints
    forward = kalman_filter_with_confidence(keypoints, confidences)
    backward = kalman_filter_with_confidence(forward[::-1], confidences[::-1])[::-1]
    return kalman_filter_without_confidence(mixed_keypoints_weighted(keypoints, confidences, backward))
