"""One-to-one, independently recalculated temporal-IoU matching."""
from collections import defaultdict


def temporal_iou(left, right):
    intersection = max(0.0, min(left["end_s"], right["end_s"]) - max(left["start_s"], right["start_s"]))
    union = max(left["end_s"], right["end_s"]) - min(left["start_s"], right["start_s"])
    return intersection / union if union else 0.0


def match_at_threshold(predictions, ground_truth, threshold):
    if not 0 < threshold < 1: raise ValueError("threshold must be between 0 and 1")
    groups = defaultdict(lambda: ([], []))
    for index, row in enumerate(predictions): groups[(row["video_id"], row["label_id"])][0].append((index, row))
    for index, row in enumerate(ground_truth): groups[(row["video_id"], row["label_id"])][1].append((index, row))
    matches, used_predictions, used_truth = [], set(), set()
    for predicted, truth in groups.values():
        candidates = [(temporal_iou(p, g), -p.get("score", 0), p["start_s"], g["start_s"], pi, gi) for pi, p in predicted for gi, g in truth]
        for iou, _, _, _, pi, gi in sorted(candidates, reverse=True):
            if iou >= threshold and pi not in used_predictions and gi not in used_truth:
                used_predictions.add(pi); used_truth.add(gi)
                matches.append({"prediction": predictions[pi], "ground_truth": ground_truth[gi], "tiou": iou})
    return {"matches": matches, "unmatched_predictions": [p for i, p in enumerate(predictions) if i not in used_predictions], "unmatched_ground_truth": [g for i, g in enumerate(ground_truth) if i not in used_truth]}
