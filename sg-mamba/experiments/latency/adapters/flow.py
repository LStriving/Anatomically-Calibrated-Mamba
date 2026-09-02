"""TV-L1 flow extraction compatible with tools/extract_flow_frame.py."""
import numpy as np

from ..contracts import StructuralRunError


class FlowExtractAdapter:
    name = "flow_extract"

    def prepare(self, context):
        try:
            import cv2
            factory = cv2.optflow.DualTVL1OpticalFlow_create
        except (ImportError, AttributeError) as error:
            raise StructuralRunError(
                "OpenCV contrib DualTVL1OpticalFlow_create is required by tools/extract_flow_frame.py"
            ) from error
        self._cv2 = cv2
        self._tvl1 = factory()

    def close(self):
        self._tvl1 = None

    def run(self, payload, context):
        frames = payload.require("frames")
        if len(frames) < 2:
            raise StructuralRunError("TV-L1 flow requires at least two decoded frames")
        flows = []
        previous = self._cv2.cvtColor(frames[0], self._cv2.COLOR_BGR2GRAY)
        for frame in frames:
            current = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2GRAY)
            flows.append(self._tvl1.calc(previous, current, None).astype(np.float32))
            previous = current
        return payload.with_value("flow", np.stack(flows))
