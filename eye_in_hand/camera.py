"""Minimal RealSense D405 camera wrapper with no project-specific deps."""

import cv2
import numpy as np
import pyrealsense2 as rs


class Camera:
    """RealSense D405 RGB-D camera."""

    def __init__(self, width=640, height=480, fps=30):
        self._width = width
        self._height = height
        self._fps = fps
        self._pipeline = None
        self._device_info = {}

    def start(self):
        cfg = rs.config()
        cfg.enable_stream(
            rs.stream.color, self._width, self._height, rs.format.bgr8, self._fps
        )
        cfg.enable_stream(
            rs.stream.depth, self._width, self._height, rs.format.z16, self._fps
        )
        self._pipeline = rs.pipeline()
        profile = self._pipeline.start(cfg)
        device = profile.get_device()
        for key, label in (
            (rs.camera_info.name, "name"),
            (rs.camera_info.serial_number, "serial_number"),
            (rs.camera_info.firmware_version, "firmware_version"),
            (rs.camera_info.usb_type_descriptor, "usb_type"),
        ):
            try:
                if device.supports(key):
                    self._device_info[label] = device.get_info(key)
            except RuntimeError:
                continue

        for _ in range(30):
            self._pipeline.wait_for_frames()

        intr = (
            profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        self._intrinsics = np.array(
            [
                [intr.fx, 0.0, intr.ppx],
                [0.0, intr.fy, intr.ppy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self._dist_coeffs = np.asarray(intr.coeffs, dtype=np.float64)
        self._distortion_model = str(intr.model)

    def stop(self):
        if self._pipeline:
            self._pipeline.stop()
            self._pipeline = None

    @property
    def intrinsics(self):
        return self._intrinsics

    @property
    def dist_coeffs(self):
        return self._dist_coeffs

    @property
    def distortion_model(self):
        return self._distortion_model

    @property
    def resolution(self):
        return self._width, self._height

    @property
    def device_info(self):
        return dict(self._device_info)

    @property
    def stream_config(self):
        return {"width": self._width, "height": self._height, "fps": self._fps}

    def capture(self):
        """Return BGR image as an HxWx3 uint8 array."""
        frames = self._pipeline.wait_for_frames()
        color = frames.get_color_frame()
        return np.asanyarray(color.get_data())


if __name__ == "__main__":
    cam = Camera()
    try:
        cam.start()
        print("RealSense camera started")
        print(f"Device info: {cam.device_info}")
        print(f"Stream config: {cam.stream_config}")
        print("Color intrinsics:")
        print(cam.intrinsics)
        print(f"Distortion model: {cam.distortion_model}")
        print(f"Distortion coefficients: {cam.dist_coeffs.tolist()}")

        img = cam.capture()
        output_path = "captured_image.jpg"
        ok = cv2.imwrite(output_path, img)
        if not ok:
            raise RuntimeError(f"Could not save image: {output_path}")
        print(f"Captured image shape: {img.shape}, dtype: {img.dtype}")
        print(f"Image saved: {output_path}")
    finally:
        cam.stop()
