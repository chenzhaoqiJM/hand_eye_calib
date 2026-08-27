"""RealSense D4xx color-camera backend for eye-to-hand calibration."""

from __future__ import annotations

import cv2
import numpy as np


class RealSenseCamera:
    """Capture BGR color frames and factory intrinsics via pyrealsense2."""

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        serial_number: str | None = None,
    ) -> None:
        if int(width) <= 0 or int(height) <= 0 or int(fps) <= 0:
            raise ValueError("width, height, and fps must be positive")
        self._width = int(width)
        self._height = int(height)
        self._fps = int(fps)
        self._serial_number = None if serial_number is None else str(serial_number)
        self._pipeline = None
        self._intrinsics = None
        self._dist_coeffs = None
        self._distortion_model = None
        self._device_info: dict[str, str] = {}
        self._stream_config: dict[str, object] = {}

    def start(self) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError(
                "RealSense support requires pyrealsense2; install "
                "requirements-realsense.txt"
            ) from exc

        config = rs.config()
        if self._serial_number:
            config.enable_device(self._serial_number)
        config.enable_stream(
            rs.stream.color,
            self._width,
            self._height,
            rs.format.bgr8,
            self._fps,
        )

        pipeline = rs.pipeline()
        try:
            try:
                profile = pipeline.start(config)
            except RuntimeError as exc:
                raise RuntimeError(
                    "Could not start the RealSense color stream; check the "
                    "device, USB connection, and requested resolution/fps"
                ) from exc
            device = profile.get_device()
            self._device_info = self._read_device_info(device, rs)
            color_profile = (
                profile.get_stream(rs.stream.color)
                .as_video_stream_profile()
            )
            intrinsics = color_profile.get_intrinsics()
            self._intrinsics = np.array(
                [
                    [intrinsics.fx, 0.0, intrinsics.ppx],
                    [0.0, intrinsics.fy, intrinsics.ppy],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            self._dist_coeffs = np.asarray(intrinsics.coeffs, dtype=np.float64)
            self._distortion_model = self._normalise_distortion_model(
                intrinsics.model
            )
            self._stream_config = {
                "backend": "RealSense",
                "serial_number": self._device_info.get("serial_number"),
                "stream": "color",
                "width": int(intrinsics.width),
                "height": int(intrinsics.height),
                "fps": self._fps,
                "format": "bgr8",
            }
            self._pipeline = pipeline
        except BaseException:
            try:
                pipeline.stop()
            except BaseException:
                pass
            raise

        try:
            for _ in range(30):
                self._pipeline.wait_for_frames()
        except BaseException:
            self.stop()
            raise

    def stop(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            finally:
                self._pipeline = None

    def capture(self) -> np.ndarray:
        if self._pipeline is None:
            raise RuntimeError("RealSense camera has not been started")
        frames = self._pipeline.wait_for_frames()
        color = frames.get_color_frame()
        if not color:
            raise RuntimeError("RealSense did not provide a color frame")
        image = np.asanyarray(color.get_data())
        if image.ndim != 3 or image.shape[2] != 3:
            raise RuntimeError(f"Expected a BGR image, got shape {image.shape}")
        return image

    @property
    def intrinsics(self) -> np.ndarray:
        if self._intrinsics is None:
            raise RuntimeError("RealSense camera has not been started")
        return self._intrinsics.copy()

    @property
    def dist_coeffs(self) -> np.ndarray:
        if self._dist_coeffs is None:
            raise RuntimeError("RealSense camera has not been started")
        return self._dist_coeffs.copy()

    @property
    def distortion_model(self) -> str:
        if self._distortion_model is None:
            raise RuntimeError("RealSense camera has not been started")
        return self._distortion_model

    @property
    def resolution(self) -> tuple[int, int]:
        if self._stream_config:
            return int(self._stream_config["width"]), int(self._stream_config["height"])
        return self._width, self._height

    @property
    def device_info(self) -> dict[str, str]:
        return dict(self._device_info)

    @property
    def stream_config(self) -> dict[str, object]:
        return dict(self._stream_config)

    @property
    def intrinsics_payload(self) -> dict[str, object]:
        """Return factory intrinsics in the project's JSON format."""
        width, height = self.resolution
        return {
            "fx": float(self.intrinsics[0, 0]),
            "fy": float(self.intrinsics[1, 1]),
            "cx": float(self.intrinsics[0, 2]),
            "cy": float(self.intrinsics[1, 2]),
            "width": width,
            "height": height,
            "coeffs": self.dist_coeffs.tolist(),
            "distortion_model": self.distortion_model,
        }

    @staticmethod
    def _read_device_info(device, rs) -> dict[str, str]:
        values: dict[str, str] = {}
        for key, label in (
            (rs.camera_info.name, "name"),
            (rs.camera_info.serial_number, "serial_number"),
            (rs.camera_info.firmware_version, "firmware_version"),
            (rs.camera_info.usb_type_descriptor, "usb_type"),
        ):
            try:
                if device.supports(key):
                    values[label] = device.get_info(key)
            except RuntimeError:
                continue
        return values

    @staticmethod
    def _normalise_distortion_model(model) -> str:
        name = str(model).lower()
        if "inverse_brown_conrady" in name:
            return "inverse_brown_conrady"
        if "brown_conrady" in name:
            return "brown_conrady"
        if name.endswith(".none") or name == "none":
            return "none"
        return name.rsplit(".", 1)[-1]


if __name__ == "__main__":
    camera = RealSenseCamera()
    try:
        camera.start()
        print("RealSense color camera started")
        print(f"Device info: {camera.device_info}")
        print(f"Stream config: {camera.stream_config}")
        print("Color intrinsics:")
        print(camera.intrinsics)
        print(f"Distortion model: {camera.distortion_model}")
        print(f"Distortion coefficients: {camera.dist_coeffs.tolist()}")
        image = camera.capture()
        output_path = "captured_realsense_color.jpg"
        if not cv2.imwrite(output_path, image):
            raise RuntimeError(f"Could not save image: {output_path}")
        print(f"Captured image shape: {image.shape}, dtype: {image.dtype}")
        print(f"Image saved: {output_path}")
    finally:
        camera.stop()