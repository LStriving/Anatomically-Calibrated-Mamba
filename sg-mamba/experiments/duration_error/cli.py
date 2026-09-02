import argparse
import json
from .analyze import run_analysis

def main(argv=None):
    parser = argparse.ArgumentParser(description="Matched duration-error analysis")
    parser.add_argument("--predictions", required=True); parser.add_argument("--annotations", required=True)
    parser.add_argument("--split", required=True); parser.add_argument("--label-map", required=True)
    parser.add_argument("--tiou-thresholds", default="0.3,0.5,0.7"); parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    label_map = {int(key): value for key, value in json.loads(open(args.label_map, encoding="utf-8").read()).items()}
    return run_analysis(args.predictions, args.annotations, args.split, label_map, args.output_dir, tuple(map(float, args.tiou_thresholds.split(","))))

if __name__ == "__main__": main()
