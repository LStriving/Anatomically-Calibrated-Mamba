"""Strict readers for evaluator output and ActivityNet-style annotations."""
import json
import pickle
from pathlib import Path


REQUIRED = ("video-id", "t-start", "t-end", "score", "label")


def load_predictions(path, label_map):
    with Path(path).open("rb") as handle: raw = pickle.load(handle)
    for key in REQUIRED:
        if key not in raw: raise ValueError("prediction result is missing {}".format(key))
    lengths = {len(raw[key]) for key in REQUIRED}
    if len(lengths) != 1: raise ValueError("prediction arrays must have equal length")
    rows = []
    for video, start, end, score, label in zip(*(raw[key] for key in REQUIRED)):
        label = int(label)
        if label not in label_map: raise ValueError("unknown prediction label {}".format(label))
        if not float(start) < float(end): raise ValueError(f"invalid prediction segment: {video, start, end}")
        rows.append({"video_id": str(video), "label_id": label, "label_name": label_map[label], "start_s": float(start), "end_s": float(end), "score": float(score)})
    return rows


def load_ground_truth(path, split, label_map):
    raw = json.loads(Path(path).read_text(encoding="utf-8")); database = raw.get("database", raw)
    rows = []
    for video, entry in database.items():
        if entry.get("subset", "").lower() != split.lower(): continue
        duration = float(entry.get("duration", 0))
        for annotation in entry.get("annotations", []):
            label = int(annotation["label_id"])
            start, end = map(float, annotation["segment"])
            if label not in label_map or annotation.get("label") != label_map[label]: raise ValueError(f"unknown or mismatched annotation label {label}")
            if not start < end <= duration: continue
            rows.append({"video_id": str(video), "label_id": label, "label_name": label_map[label], "start_s": start, "end_s": end})
    return rows
