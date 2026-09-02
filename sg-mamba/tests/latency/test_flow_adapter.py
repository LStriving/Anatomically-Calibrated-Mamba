import pytest

from experiments.latency.adapters.flow import FlowExtractAdapter
from experiments.latency.contracts import Payload, StructuralRunError


def test_flow_adapter_requires_the_project_tvl1_factory():
    """Replacing unavailable TV-L1 with a different algorithm would invalidate comparability."""
    adapter = FlowExtractAdapter()
    with pytest.raises(StructuralRunError, match="DualTVL1"):
        adapter.prepare({})
