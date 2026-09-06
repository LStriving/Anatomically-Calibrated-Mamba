"""Detector boundaries and evaluator-compatible prediction post-processing."""
from copy import deepcopy

from ..contracts import StructuralRunError


class CoarseDetectorAdapter:
    name = "coarse_detector"

    def __init__(self, predictor=None):
        self.predictor = predictor

    def prepare(self, context):
        if self.predictor is None:
            raise StructuralRunError("Coarse detector factory is required; configure the stage-1 evaluator model")

    def close(self):
        return None

    def run(self, payload, context):
        return payload.with_value("coarse_segments", self.predictor(payload, context))


class FineDetectorAdapter:
    name = "fine_detector"

    def __init__(self, predictor=None):
        self.predictor = predictor

    def prepare(self, context):
        if self.predictor is None:
            raise StructuralRunError("Fine detector factory is required; configure the eval2tower two-tower model")

    def close(self):
        return None

    def run(self, payload, context):
        return payload.with_value("fine_predictions", self.predictor(payload, context))


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
