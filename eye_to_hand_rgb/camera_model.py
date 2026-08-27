"""Camera-model helpers for V4L2 and RealSense RGB cameras."""

from __future__ import annotations

import cv2
import numpy as np


def camera_matrix(intrinsics: dict) -> np.ndarray:
    cx = intrinsics.get("cx", intrinsics.get("ppx"))
    cy = intrinsics.get("cy", intrinsics.get("ppy"))
    if cx is None or cy is None:
        raise ValueError("Camera intrinsics require cx/cy or ppx/ppy")
    return np.array(
        [
            [intrinsics["fx"], 0.0, cx],
            [0.0, intrinsics["fy"], cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def distortion_coefficients(intrinsics: dict) -> np.ndarray | None:
    coefficients = np.asarray(intrinsics.get("coeffs", []), dtype=np.float64)
    return coefficients if coefficients.size else None


def validate_intrinsics(intrinsics: dict, width: int | None = None, height: int | None = None) -> None:
    required = ("fx", "fy", "width", "height")
    missing = [key for key in required if key not in intrinsics]
    if "cx" not in intrinsics and "ppx" not in intrinsics:
        missing.append("cx (or ppx)")
    if "cy" not in intrinsics and "ppy" not in intrinsics:
        missing.append("cy (or ppy)")
    if missing:
        raise ValueError(f"Missing camera intrinsic fields: {', '.join(missing)}")
    values = np.asarray(
        [intrinsics["fx"], intrinsics["fy"], camera_matrix(intrinsics)[0, 2],
         camera_matrix(intrinsics)[1, 2], intrinsics["width"], intrinsics["height"]],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Camera intrinsic values must be finite and positive")
    model = _model(intrinsics)
    if model not in ("plumb_bob", "brown_conrady", "inverse_brown_conrady", "opencv", "none"):
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
    """Solve PnP without applying OpenCV's model to inverse Brown pixels."""
    validate_intrinsics(intrinsics)
    model = _model(intrinsics)
    coefficients = _coefficients(intrinsics)
    has_distortion = bool(
        coefficients.size and np.any(np.abs(coefficients) > 1e-12)
    )
    if has_distortion and model == "inverse_brown_conrady":
        normalized = _inverse_brown_normalized(image_points, intrinsics)
        return cv2.solvePnP(
            np.asarray(object_points, dtype=np.float64),
            normalized,
            np.eye(3, dtype=np.float64),
            None,
            flags=flags,
        )
    if has_distortion and model not in ("plumb_bob", "brown_conrady", "opencv"):
        raise ValueError(f"Unsupported distortion_model: {model}")
    return cv2.solvePnP(
        np.asarray(object_points, dtype=np.float64),
        np.asarray(image_points, dtype=np.float64),
        camera_matrix(intrinsics),
        coefficients if has_distortion else None,
        flags=flags,
    )


def reprojection_rms_px(
    object_points, image_points, rvec, tvec, intrinsics: dict
) -> float:
    model = _model(intrinsics)
    coefficients = _coefficients(intrinsics)
    has_distortion = bool(
        coefficients.size and np.any(np.abs(coefficients) > 1e-12)
    )
    if has_distortion and model == "inverse_brown_conrady":
        rotation = cv2.Rodrigues(rvec)[0]
        camera_points = (
            rotation @ np.asarray(object_points, dtype=np.float64).T
            + np.asarray(tvec, dtype=np.float64).reshape(3, 1)
        ).T
        predicted = camera_points[:, :2] / camera_points[:, 2:3]
        observed = _inverse_brown_normalized(image_points, intrinsics)
        scale = np.array(
            [intrinsics["fx"], intrinsics["fy"]], dtype=np.float64
        )
        return float(np.sqrt(np.mean(((predicted - observed) * scale) ** 2)))

    projected, _ = cv2.projectPoints(
        np.asarray(object_points, dtype=np.float64),
        rvec,
        tvec,
        camera_matrix(intrinsics),
        coefficients if has_distortion else None,
    )
    residual = projected.reshape(-1, 2) - np.asarray(image_points).reshape(-1, 2)
    return float(np.sqrt(np.mean(residual**2)))


def _coefficients(intrinsics: dict) -> np.ndarray:
    return np.asarray(intrinsics.get("coeffs", []), dtype=np.float64)


def _model(intrinsics: dict) -> str:
    model = str(intrinsics.get("distortion_model", "plumb_bob")).lower()
    if "inverse_brown_conrady" in model:
        return "inverse_brown_conrady"
    if "brown_conrady" in model:
        return "brown_conrady"
    if model.endswith(".none") or model == "none":
        return "none"
    return model


def _inverse_brown_normalized(
    image_points: np.ndarray, intrinsics: dict
) -> np.ndarray:
    """Convert inverse-Brown image pixels to normalized pinhole coordinates."""
    try:
        import pyrealsense2 as rs  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "inverse Brown-Conrady calibration needs pyrealsense2"
        ) from exc

    matrix = camera_matrix(intrinsics)
    rs_intrinsics = rs.intrinsics()
    rs_intrinsics.width = int(intrinsics["width"])
    rs_intrinsics.height = int(intrinsics["height"])
    rs_intrinsics.fx = float(matrix[0, 0])
    rs_intrinsics.fy = float(matrix[1, 1])
    rs_intrinsics.ppx = float(matrix[0, 2])
    rs_intrinsics.ppy = float(matrix[1, 2])
    rs_intrinsics.model = rs.distortion.inverse_brown_conrady
    coefficients = _coefficients(intrinsics)
    rs_intrinsics.coeffs = np.pad(
        coefficients[:5], (0, max(0, 5 - coefficients.size))
    ).tolist()
    rays = [
        rs.rs2_deproject_pixel_to_point(
            rs_intrinsics, [float(point[0]), float(point[1])], 1.0
        )
        for point in np.asarray(image_points).reshape(-1, 2)
    ]
    return np.ascontiguousarray(np.asarray(rays, dtype=np.float64)[:, :2])