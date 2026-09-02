from pathlib import Path

import pytest

from experiments.latency.adapters.video import VideoDecodeAdapter
from experiments.latency.contracts import Payload


REAL_VIDEO = Path(r"D:\LYR\lab\吞咽造影\2_2021_02_01_clip1_32s.avi")


@pytest.mark.skipif(not REAL_VIDEO.is_file(), reason="local VFSS video is unavailable")
def test_video_adapter_decodes_real_vfss_video():
    """Removing OpenCV decoding would make a valid local VFSS input unusable."""
    adapter = VideoDecodeAdapter()
    payload = adapter.run(Payload({"video": {"id": "vfss", "path": str(REAL_VIDEO)}}), {})
    frames = payload.require("frames")
    assert len(frames) > 1
    assert payload.require("fps") > 0
