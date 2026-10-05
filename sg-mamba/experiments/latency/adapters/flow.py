"""TV-L1 flow extraction compatible with tools/extract_flow_frame.py."""
import numpy as np

from ..contracts import StructuralRunError


class FlowExtractAdapter:
    name = "flow_extract"

    def __init__(self, width=128, height=128, use_gpu=False):
        self.width = width
        self.height = height
        self.use_gpu = use_gpu

    def prepare(self, context):
        try:
            import cv2
        except ImportError as error:
            raise StructuralRunError(
                "OpenCV contrib DualTVL1OpticalFlow_create is required by tools/extract_flow_frame.py"
            ) from error
        self._cv2 = cv2
        if self.use_gpu:
            if not hasattr(cv2, "cuda") or cv2.cuda.getCudaEnabledDeviceCount() < 1:
                raise StructuralRunError("GPU flow extraction requires an OpenCV CUDA build and a visible CUDA device")
            factory = getattr(cv2, "cuda_OpticalFlowDual_TVL1_create", None)
            if factory is None:
                factory = getattr(getattr(cv2, "cuda", None), "OpticalFlowDual_TVL1_create", None)
            if factory is None:
                raise StructuralRunError("OpenCV CUDA TV-L1 optical flow factory is unavailable")
            self._tvl1 = factory()
            self._gpu_prev = cv2.cuda_GpuMat()
            self._gpu_current = cv2.cuda_GpuMat()
        else:
            try:
                factory = cv2.optflow.DualTVL1OpticalFlow_create
            except AttributeError as error:
                raise StructuralRunError(
                    "OpenCV contrib DualTVL1OpticalFlow_create is required by tools/extract_flow_frame.py"
                ) from error
            self._tvl1 = factory()

    def close(self):
        self._tvl1 = None
        self._gpu_prev = None
        self._gpu_current = None

    def run(self, payload, context):
        frames = payload.require("frames")
        if len(frames) < 2:
            raise StructuralRunError("TV-L1 flow requires at least two decoded frames")
        flows = []
        previous = self._gray(frames[0])
        if self.use_gpu:
            self._gpu_prev.upload(previous)
        for frame in frames:
            current = self._gray(frame)
            if self.use_gpu:
                self._gpu_current.upload(current)
                flow = self._tvl1.calc(self._gpu_prev, self._gpu_current, None).download()
                self._gpu_prev, self._gpu_current = self._gpu_current, self._gpu_prev
            else:
                flow = self._tvl1.calc(previous, current, None)
            flows.append(flow.astype(np.float32))
            previous = current
        return payload.with_value("flow", np.stack(flows))

    def _gray(self, frame):
        resized = self._cv2.resize(frame, (self.width, self.height))
        return self._cv2.cvtColor(resized, self._cv2.COLOR_BGR2GRAY)
