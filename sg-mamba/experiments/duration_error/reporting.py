"""Duration and boundary agreement values for matched segments."""
import math


def pair_metrics(prediction, ground_truth):
    predicted_duration = prediction["end_s"] - prediction["start_s"]
    truth_duration = ground_truth["end_s"] - ground_truth["start_s"]
    duration = (predicted_duration - truth_duration) * 1000
    return {"duration_error_ms": duration, "relative_error_pct": 100 * duration / (truth_duration * 1000) if truth_duration > 0 else None, "onset_error_ms": (prediction["start_s"] - ground_truth["start_s"]) * 1000, "offset_error_ms": (prediction["end_s"] - ground_truth["end_s"]) * 1000}
