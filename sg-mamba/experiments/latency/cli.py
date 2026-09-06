"""Config and manifest entry point for auditable latency runs."""
import argparse
import importlib
import json
from pathlib import Path

import yaml

from .adapters.flow import FlowExtractAdapter
from .adapters.video import VideoDecodeAdapter
from .contracts import StructuralRunError
from .metadata import write_meta_report
from .runner import run_benchmark


def load_config(path):
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise StructuralRunError("Latency config must be a mapping")
    return config


def load_manifest(path):
    with Path(path).open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if isinstance(manifest, dict):
        manifest = manifest.get("videos")
    if not isinstance(manifest, list):
        raise StructuralRunError("Manifest must be a JSON list or an object with a videos list")
    for video in manifest:
        if not isinstance(video, dict) or not isinstance(video.get("id"), str) or not isinstance(video.get("path"), str):
            raise StructuralRunError("Each manifest video requires string id and path fields")
    return manifest


def build_adapters(config):
    adapters = []
    for stage in config.get("stages", []):
        if isinstance(stage, dict):
            adapters.append(_build_factory_stage(stage, config.get("weights_mode")))
            continue
        if stage == "decode":
            adapters.append(VideoDecodeAdapter(max_frames=config.get("max_frames")))
        elif stage == "flow_extract":
            adapters.append(FlowExtractAdapter(
                width=config.get("flow_width", 128), height=config.get("flow_height", 128)
            ))
        else:
            raise StructuralRunError(
                "Stage {} requires an explicit model factory and is not available in the basic CLI".format(stage)
            )
    if not adapters:
        raise StructuralRunError("Latency config must declare at least one stage")
    return adapters


def _build_factory_stage(stage, weights_mode=None):
    name = stage.get("name")
    factory_path = stage.get("factory")
    kwargs = stage.get("kwargs", {})
    if not isinstance(name, str) or not isinstance(factory_path, str) or not isinstance(kwargs, dict):
        raise StructuralRunError("Configured stage requires string name, string factory, and mapping kwargs")
    kwargs = dict(kwargs)
    if weights_mode is not None and "weights_mode" in kwargs:
        kwargs["weights_mode"] = weights_mode
    try:
        module_name, attribute = factory_path.split(":", 1)
        factory = getattr(importlib.import_module(module_name), attribute)
    except (ValueError, ImportError, AttributeError) as error:
        raise StructuralRunError("Unable to import stage factory: {}".format(factory_path)) from error
    try:
        adapter = factory(**kwargs)
    except TypeError as error:
        raise StructuralRunError("Unable to construct stage {} from {}".format(name, factory_path)) from error
    if getattr(adapter, "name", None) != name:
        raise StructuralRunError("Stage factory {} produced {} rather than {}".format(
            factory_path, getattr(adapter, "name", None), name
        ))
    return adapter


def main(argv=None):
    parser = argparse.ArgumentParser(description="Auditable raw-video latency benchmark")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("validate", "smoke", "benchmark"))
    parser.add_argument("--weights-mode", choices=("skip", "required"))
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.mode is not None:
        config["mode"] = args.mode
    if args.weights_mode is not None:
        config["weights_mode"] = args.weights_mode
    try:
        manifest = load_manifest(args.manifest)
        adapters = build_adapters(config)
    except StructuralRunError as error:
        write_meta_report(args.output_dir, {
            "args": config,
            "scope": "validation" if config.get("mode") == "validate" else "partial_pipeline",
            "successes": [],
            "failures": [],
            "structural_failure": {"message": str(error), "type": type(error).__name__},
            "artifacts": [],
        })
        raise
    return run_benchmark(config, manifest, adapters, args.output_dir)


if __name__ == "__main__":
    print(main())
