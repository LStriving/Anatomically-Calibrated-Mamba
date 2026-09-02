import csv
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
    for threshold in thresholds:
        for match in match_at_threshold(predicted, truth, threshold)["matches"]:
            rows.append(dict(tiou_threshold=threshold, tiou=match["tiou"], **match["prediction"], **pair_metrics(match["prediction"], match["ground_truth"])))
    with pairs_path.open("w", newline="", encoding="utf-8") as handle:
        writer=csv.DictWriter(handle, fieldnames=sorted({k for row in rows for k in row})); writer.writeheader(); writer.writerows(rows)
    summary_path = output_dir / ("duration_summary_" + stamp + ".csv")
    summaries = []
    for threshold in thresholds:
        selected = [row for row in rows if row["tiou_threshold"] == threshold]
        summaries.append({"tiou_threshold": threshold, "matched_count": len(selected), "gt_count": len(truth), "prediction_count": len(predicted), "match_rate": len(selected) / len(truth) if truth else 0.0})
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
    limitations_path = output_dir / ("duration_limitations_" + stamp + ".md")
    limitations_path.write_text("# Limitations\n\nDuration error is matched-boundary agreement with existing ground truth. It is conditional on tIoU and does not replace mAP, which ranks confidence and includes false positives, false negatives, classification, and localization.\n", encoding="utf-8")
    return write_meta_report(output_dir, {"scope":"validation", "artifacts":[{"path":pairs_path,"purpose":"duration matched pairs"}, {"path":summary_path,"purpose":"duration threshold summary"}, {"path":limitations_path,"purpose":"factual limitations"}]})
