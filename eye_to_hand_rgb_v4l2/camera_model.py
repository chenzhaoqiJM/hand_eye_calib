"""Pinhole camera-model helpers for calibrated RGB cameras."""

from __future__ import annotations

import cv2
import numpy as np


def camera_matrix(intrinsics: dict) -> np.ndarray:
    return np.array(
        [
            [intrinsics["fx"], 0.0, intrinsics["cx"]],
            [0.0, intrinsics["fy"], intrinsics["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def distortion_coefficients(intrinsics: dict) -> np.ndarray | None:
    coefficients = np.asarray(intrinsics.get("coeffs", []), dtype=np.float64)
    return coefficients if coefficients.size else None


def validate_intrinsics(intrinsics: dict, width: int | None = None, height: int | None = None) -> None:
    required = ("fx", "fy", "cx", "cy", "width", "height")
    missing = [key for key in required if key not in intrinsics]
    if missing:
        raise ValueError(f"Missing camera intrinsic fields: {', '.join(missing)}")
    values = np.asarray([intrinsics[key] for key in required], dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Camera intrinsic values must be finite and positive")
    model = str(intrinsics.get("distortion_model", "plumb_bob")).lower()
    if model not in ("plumb_bob", "brown_conrady", "opencv", "none"):
        raise ValueError(f"Unsupported distortion_model: {model}")
    if width is not None and int(intrinsics["width"]) != int(width):
        raise ValueError(
            f"Intrinsic width {intrinsics['width']} does not match camera width {width}"
        )
    if height is not None and int(intrinsics["height"]) != int(height):
        raise ValueError(
            f"Intrinsic height {intrinsics['height']} does not match camera height {height}"
        )


def solve_pnp(object_points, image_points, intrinsics: dict, flags: int):
    validate_intrinsics(intrinsics)
    return cv2.solvePnP(
        np.asarray(object_points, dtype=np.float64),
        np.asarray(image_points, dtype=np.float64),
        camera_matrix(intrinsics),
        distortion_coefficients(intrinsics),
        flags=flags,
    )


def reprojection_rms_px(
    object_points, image_points, rvec, tvec, intrinsics: dict
) -> float:
    projected, _ = cv2.projectPoints(
        np.asarray(object_points, dtype=np.float64),
        rvec,
        tvec,
        camera_matrix(intrinsics),
        distortion_coefficients(intrinsics),
    )
    residual = projected.reshape(-1, 2) - np.asarray(image_points).reshape(-1, 2)
    return float(np.sqrt(np.mean(residual**2)))