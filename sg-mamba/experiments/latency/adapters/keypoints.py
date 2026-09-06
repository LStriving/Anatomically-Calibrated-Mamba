"""Frame-array keypoint and Kalman stages using the project's NPY API."""
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
        return payload.with_value(
            "skeleton_features",
            self.encoder(payload.require("smoothed_keypoints"), payload.require("keypoint_confidences")),
        )


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
