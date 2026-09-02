"""OpenCV video decoding without changing process-wide paths or state."""
from ..contracts import RecoverableVideoError


class VideoDecodeAdapter:
    name = "decode"

    def prepare(self, context):
        return None

    def close(self):
        return None

    def run(self, payload, context):
        try:
            import cv2
        except ImportError as error:
            raise RecoverableVideoError("OpenCV is unavailable") from error
        video = payload.require("video")
        capture = cv2.VideoCapture(video["path"])
        if not capture.isOpened():
            raise RecoverableVideoError("unable to open video: {}".format(video["path"]))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        capture.release()
        if not frames or fps <= 0:
            raise RecoverableVideoError("video has no decodable frames or fps")
        return payload.with_value("frames", frames).with_value("fps", fps)
