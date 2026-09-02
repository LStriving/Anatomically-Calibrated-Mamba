import numpy as np

from experiments.latency.adapters.flow import FlowExtractAdapter
from experiments.latency.contracts import Payload


def test_flow_adapter_produces_tvl1_flow_for_decoded_frames():
    """Removing TV-L1 extraction would leave downstream flow features absent."""
    adapter = FlowExtractAdapter()
    adapter.prepare({})
    try:
        frames = [np.zeros((16, 16, 3), dtype=np.uint8), np.ones((16, 16, 3), dtype=np.uint8)]
        flow = adapter.run(Payload({"frames": frames}), {}).require("flow")
        assert flow.shape == (2, 128, 128, 2)
        assert flow.dtype == np.float32
    finally:
        adapter.close()
