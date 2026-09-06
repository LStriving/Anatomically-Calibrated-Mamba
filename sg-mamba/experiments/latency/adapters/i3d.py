"""I3D feature extraction compatible with ``pytorch_i3d.InceptionI3d``."""
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
