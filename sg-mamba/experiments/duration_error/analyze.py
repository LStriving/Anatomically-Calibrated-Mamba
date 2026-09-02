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
    return write_meta_report(output_dir, {"scope":"validation", "artifacts":[{"path":pairs_path,"purpose":"duration matched pairs"}]})
