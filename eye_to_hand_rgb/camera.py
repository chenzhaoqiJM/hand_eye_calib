"""RGB camera wrapper using a standard Linux V4L2 /dev/video node."""

from __future__ import annotations

import re
import time

import cv2
import numpy as np


class Camera:
    """Capture BGR frames through OpenCV's V4L2 backend."""

    def __init__(
        self,
        device: str = "/dev/video0",
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        pixel_format: str = "MJPG",
    ) -> None:
        if not re.fullmatch(r"/dev/video\d+", device):
            raise ValueError("device must be a /dev/videoN node")
        if len(pixel_format) != 4 or not pixel_format.isascii():
            raise ValueError("pixel_format must be a four-character ASCII FourCC")
        self._device = device
        self._width = int(width)
        self._height = int(height)
        self._fps = int(fps)
        self._pixel_format = pixel_format.upper()
        self._capture: cv2.VideoCapture | None = None
        self._stream_config: dict = {}

    def start(self) -> None:
        capture = cv2.VideoCapture(self._device, cv2.CAP_V4L2)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Could not open V4L2 camera: {self._device}")

        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self._pixel_format))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        capture.set(cv2.CAP_PROP_FPS, self._fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._capture = capture

        for _ in range(10):
            ok, _ = capture.read()
            if not ok:
                self.stop()
                raise RuntimeError(f"Could not read frames from {self._device}")
            time.sleep(0.01)

        actual_fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
        self._stream_config = {
            "device": self._device,
            "backend": capture.getBackendName(),
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "pixel_format": "".join(
                chr((actual_fourcc >> (8 * offset)) & 0xFF) for offset in range(4)
            ),
        }

    def stop(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def capture(self) -> np.ndarray:
        if self._capture is None:
            raise RuntimeError("Camera has not been started")
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to capture a frame from {self._device}")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise RuntimeError(f"Expected a BGR image, got shape {frame.shape}")
        return frame

    @property
    def resolution(self) -> tuple[int, int]:
        if self._stream_config:
            return int(self._stream_config["width"]), int(self._stream_config["height"])
        return self._width, self._height

    @property
    def device_info(self) -> dict:
        return {"node": self._device, "api": "V4L2"}

    @property
    def stream_config(self) -> dict:
        return dict(self._stream_config)