"""OpenCV V4L2 camera wrapper used by the calibration application."""

from __future__ import annotations

import re
import time

import cv2
import numpy as np


class V4L2Camera:
    """Capture BGR frames from a Linux /dev/videoN node."""

    def __init__(
        self,
        device: str,
        width: int,
        height: int,
        fps: int,
        pixel_format: str,
    ) -> None:
        if not re.fullmatch(r"/dev/video\d+", device):
            raise ValueError("device must be a /dev/videoN node")
        if width <= 0 or height <= 0 or fps <= 0:
            raise ValueError("width, height and fps must be positive")
        if len(pixel_format) != 4 or not pixel_format.isascii():
            raise ValueError("pixel format must be a four-character ASCII FourCC")
        self._device = device
        self._width = int(width)
        self._height = int(height)
        self._fps = int(fps)
        self._pixel_format = pixel_format.upper()
        self._capture: cv2.VideoCapture | None = None
        self._config: dict[str, object] = {}

    def start(self) -> None:
        capture = cv2.VideoCapture(self._device, cv2.CAP_V4L2)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"cannot open V4L2 camera: {self._device}")
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
                raise RuntimeError(f"cannot read from V4L2 camera: {self._device}")
            time.sleep(0.01)
        fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
        self._config = {
            "device": self._device,
            "backend": capture.getBackendName(),
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "pixel_format": "".join(
                chr((fourcc >> (8 * offset)) & 0xFF) for offset in range(4)
            ),
        }

    def read(self) -> np.ndarray:
        if self._capture is None:
            raise RuntimeError("camera is not started")
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"failed to capture frame from {self._device}")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise RuntimeError(f"expected BGR frame, got shape {frame.shape}")
        return frame

    def stop(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    @property
    def config(self) -> dict[str, object]:
        return dict(self._config)
