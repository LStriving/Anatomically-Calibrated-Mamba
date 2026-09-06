"""I3D feature extraction compatible with ``pytorch_i3d.InceptionI3d``."""
from collections import OrderedDict
from pathlib import Path
import warnings

import numpy as np

from ..contracts import StructuralRunError


class I3DExtractAdapter:
    """Extract RGB and TV-L1 features with the project's eight-frame I3D input."""

    name = "i3d_extract"

    def __init__(self, rgb_model=None, flow_model=None, image_size=128, window_size=8, window_step=1, device=None):
        self.rgb_model = rgb_model
        self.flow_model = flow_model
        self.image_size = image_size
        self.window_size = window_size
        self.window_step = window_step
        self.device = device
        self.weights_loaded = None
        self.checkpoint_warnings = []

    def prepare(self, context):
        if self.rgb_model is None or self.flow_model is None:
            raise StructuralRunError(
                "I3D models must be supplied by an explicit factory; use pytorch_i3d.InceptionI3d for RGB and flow"
            )
        for model in (self.rgb_model, self.flow_model):
            if hasattr(model, "eval"):
                model.eval()

    def close(self):
        return None

    def run(self, payload, context):
        import torch
        import torch.nn.functional as functional

        frames = np.asarray(payload.require("frames"))
        flow = np.asarray(payload.require("flow"))
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise StructuralRunError("I3D RGB input must have shape (frames, height, width, 3)")
        if flow.ndim != 4 or flow.shape[-1] != 2:
            raise StructuralRunError("I3D flow input must have shape (frames, height, width, 2)")
        if len(frames) != len(flow):
            raise StructuralRunError("I3D RGB and flow frame counts must match")
        rgb = torch.from_numpy(frames).float().permute(0, 3, 1, 2)
        rgb = functional.interpolate(rgb, (self.image_size, self.image_size), mode="bilinear", align_corners=False)
        rgb = (rgb / 127.5) - 1.0
        temporal = torch.from_numpy(flow).float().permute(0, 3, 1, 2)
        temporal = functional.interpolate(temporal, (self.image_size, self.image_size), mode="bilinear", align_corners=False)
        rgb_windows = self._windows(rgb).permute(0, 2, 1, 3, 4)
        flow_windows = self._windows(temporal).permute(0, 2, 1, 3, 4)
        if self.device is not None:
            rgb_windows = rgb_windows.to(self.device)
            flow_windows = flow_windows.to(self.device)
        with torch.no_grad():
            rgb_features = self.rgb_model.extract_features(rgb_windows)
            flow_features = self.flow_model.extract_features(flow_windows)
        rgb_features = rgb_features[:, :, 0, 0, 0].detach().cpu().numpy()
        flow_features = flow_features[:, :, 0, 0, 0].detach().cpu().numpy()
        return payload.with_value("rgb_i3d_features", rgb_features).with_value("flow_i3d_features", flow_features)

    def _windows(self, values):
        if len(values) < self.window_size:
            raise StructuralRunError("I3D requires at least {} frames".format(self.window_size))
        windows = [values[start:start + self.window_size] for start in range(0, len(values) - self.window_size + 1, self.window_step)]
        return __import__("torch").stack(windows)


def make_i3d_adapter(rgb_checkpoint, flow_checkpoint, weights_mode="required", device=None, **kwargs):
    """Construct project-standard RGB/flow I3D models outside per-video timing."""
    import torch
    from pytorch_i3d import InceptionI3d

    paths = {"rgb": Path(rgb_checkpoint), "flow": Path(flow_checkpoint)}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing and weights_mode == "required":
        raise StructuralRunError("I3D checkpoint does not exist: {}".format(
            ", ".join(str(paths[name]) for name in missing)
        ))
    target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rgb_model = InceptionI3d(7, in_channels=3)
    flow_model = InceptionI3d(400, in_channels=2)
    adapter = I3DExtractAdapter(rgb_model=rgb_model.to(target), flow_model=flow_model.to(target), device=target, **kwargs)
    adapter.weights_loaded = not missing
    if missing:
        adapter.checkpoint_warnings.append("I3D random weights used because checkpoint is unavailable: {}".format(
            ", ".join(missing)
        ))
        warnings.warn(adapter.checkpoint_warnings[-1], RuntimeWarning)
        return adapter
    try:
        flow_model.load_state_dict(_state_dict(torch.load(paths["flow"], map_location=target)))
        rgb_model.load_state_dict(_strip_module_prefix(_state_dict(torch.load(paths["rgb"], map_location=target))))
    except (RuntimeError, KeyError, TypeError, OSError) as error:
        if weights_mode == "required":
            raise StructuralRunError("Unable to load required I3D checkpoint") from error
        adapter.weights_loaded = False
        adapter.checkpoint_warnings.append("I3D random weights used after checkpoint load failure: {}".format(error))
        warnings.warn(adapter.checkpoint_warnings[-1], RuntimeWarning)
    return adapter


def _state_dict(checkpoint):
    return checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint


def _strip_module_prefix(state_dict):
    return OrderedDict((key.replace("module.", ""), value) for key, value in state_dict.items())
