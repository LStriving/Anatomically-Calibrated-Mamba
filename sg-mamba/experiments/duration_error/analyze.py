import csv
import math
from datetime import datetime, timezone
from pathlib import Path
from .matching import match_at_threshold
from .metadata import write_meta_report
from .reporting import pair_metrics
from .schema import load_ground_truth, load_predictions

def run_analysis(predictions, annotations, split, label_map, output_dir, thresholds=(.3,.5,.7)):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    predicted = load_predictions(predictions, label_map); truth = load_ground_truth(annotations, split, label_map)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"); pairs_path = output_dir / ("duration_pairs_" + stamp + ".csv"); rows=[]
    match_results = {}
    for threshold in thresholds:
        match_results[threshold] = match_at_threshold(predicted, truth, threshold)
        for match in match_results[threshold]["matches"]:
            rows.append(dict(tiou_threshold=threshold, tiou=match["tiou"], **match["prediction"], **pair_metrics(match["prediction"], match["ground_truth"])))
    with pairs_path.open("w", newline="", encoding="utf-8") as handle:
        pair_fields = sorted({k for row in rows for k in row}) or [
            "tiou_threshold", "tiou", "video_id", "label_id", "label_name", "start_s",
            "end_s", "score", "duration_error_ms", "relative_error_pct",
            "onset_error_ms", "offset_error_ms",
        ]
        writer=csv.DictWriter(handle, fieldnames=pair_fields); writer.writeheader(); writer.writerows(rows)
    summary_path = output_dir / ("duration_summary_" + stamp + ".csv")
    summaries = _summary_rows(match_results, predicted, truth, label_map)
    fields = [
        "tiou_threshold", "label_id", "label_name", "matched_count", "gt_count",
        "prediction_count", "false_positive_count", "false_negative_count",
        "match_rate",
    ]
    for metric in ("duration_error_ms", "onset_error_ms", "offset_error_ms", "relative_error_pct"):
        fields.extend([
            metric + "_mae", metric + "_rmse", metric + "_bias",
            metric + "_p10", metric + "_p50", metric + "_p90",
        ])
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(summaries)
    limitations_path = output_dir / ("duration_limitations_" + stamp + ".md")
    limitations_path.write_text("# Limitations\n\nDuration error is matched-boundary agreement with existing ground truth. It is conditional on tIoU and does not replace mAP, which ranks confidence and includes false positives, false negatives, classification, and localization.\n", encoding="utf-8")
    return write_meta_report(output_dir, {"scope":"validation", "artifacts":[{"path":pairs_path,"purpose":"duration matched pairs"}, {"path":summary_path,"purpose":"duration threshold summary"}, {"path":limitations_path,"purpose":"factual limitations"}]})


def _summary_rows(match_results, predicted, truth, label_map):
    summaries = []
    for threshold in match_results:
        result = match_results[threshold]
        for label_id in sorted(label_map):
            label_predictions = [row for row in predicted if row["label_id"] == label_id]
            label_truth = [row for row in truth if row["label_id"] == label_id]
            label_matches = [
                match for match in result["matches"]
                if match["prediction"]["label_id"] == label_id
            ]
            row = {
                "tiou_threshold": threshold,
                "label_id": label_id,
                "label_name": label_map[label_id],
                "matched_count": len(label_matches),
                "gt_count": len(label_truth),
                "prediction_count": len(label_predictions),
                "false_positive_count": len([
                    item for item in result["unmatched_predictions"]
                    if item["label_id"] == label_id
                ]),
                "false_negative_count": len([
                    item for item in result["unmatched_ground_truth"]
                    if item["label_id"] == label_id
                ]),
                "match_rate": len(label_matches) / len(label_truth) if label_truth else 0.0,
            }
            metrics = [pair_metrics(match["prediction"], match["ground_truth"]) for match in label_matches]
            for metric in ("duration_error_ms", "onset_error_ms", "offset_error_ms", "relative_error_pct"):
                row.update(_error_stats(metric, [item.get(metric) for item in metrics]))
            summaries.append(row)
    return summaries


def _error_stats(name, values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return {
            name + "_mae": "",
            name + "_rmse": "",
            name + "_bias": "",
            name + "_p10": "",
            name + "_p50": "",
            name + "_p90": "",
        }
    return {
        name + "_mae": sum(abs(value) for value in values) / len(values),
        name + "_rmse": math.sqrt(sum(value * value for value in values) / len(values)),
        name + "_bias": sum(values) / len(values),
        name + "_p10": _percentile(values, 10),
        name + "_p50": _percentile(values, 50),
        name + "_p90": _percentile(values, 90),
    }


def _percentile(values, percentile):
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * percentile / 100
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return values[lower]
    weight = rank - lower
    return values[lower] * (1 - weight) + values[upper] * weight
