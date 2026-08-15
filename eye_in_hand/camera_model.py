"""RealSense camera-model helpers shared by hand-eye solve and validation."""

from __future__ import annotations

import cv2
import numpy as np


def camera_matrix(intrinsics: dict) -> np.ndarray:
    cx = intrinsics.get("cx", intrinsics.get("ppx"))
    cy = intrinsics.get("cy", intrinsics.get("ppy"))
    return np.array(
        [
            [intrinsics["fx"], 0.0, cx],
            [0.0, intrinsics["fy"], cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _coefficients(intrinsics: dict) -> np.ndarray:
    return np.asarray(intrinsics.get("coeffs", []), dtype=np.float64)


def _model(intrinsics: dict) -> str:
    return str(intrinsics.get("distortion_model", "none")).lower()


def _inverse_brown_normalized(image_points: np.ndarray, intrinsics: dict) -> np.ndarray:
    try:
        import pyrealsense2 as rs  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("inverse Brown-Conrady calibration needs pyrealsense2") from exc

    matrix = camera_matrix(intrinsics)
    rs_intrinsics = rs.intrinsics()
    rs_intrinsics.width = int(intrinsics.get("width", 0))
    rs_intrinsics.height = int(intrinsics.get("height", 0))
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
    # The [:, :2] view has a row stride for three coordinates, which makes it
    # non-contiguous.  OpenCV's solvePnP input validation rejects that layout
    # even though its shape is (N, 2), so materialize a contiguous array.
    return np.ascontiguousarray(np.asarray(rays, dtype=np.float64)[:, :2])


def solve_pnp(object_points, image_points, intrinsics, flags):
    """Solve PnP without silently applying the wrong distortion convention."""
    model = _model(intrinsics)
    coefficients = _coefficients(intrinsics)
    has_distortion = bool(
        coefficients.size and np.any(np.abs(coefficients) > 1e-12)
    )
    if has_distortion and "inverse_brown" in model:
        normalized = _inverse_brown_normalized(image_points, intrinsics)
        return cv2.solvePnP(
            np.asarray(object_points),
            normalized,
            np.eye(3, dtype=np.float64),
            None,
            flags=flags,
        )
    if has_distortion and "brown_conrady" not in model:
        raise RuntimeError(f"Unsupported RealSense distortion model: {model}")
    return cv2.solvePnP(
        np.asarray(object_points),
        np.asarray(image_points),
        camera_matrix(intrinsics),
        coefficients if has_distortion else None,
        flags=flags,
    )


def reprojection_rms_px(object_points, image_points, rvec, tvec, intrinsics) -> float:
    model = _model(intrinsics)
    coefficients = _coefficients(intrinsics)
    has_distortion = bool(
        coefficients.size and np.any(np.abs(coefficients) > 1e-12)
    )
    if has_distortion and "inverse_brown" in model:
        rotation = cv2.Rodrigues(rvec)[0]
        camera_points = (
            rotation @ np.asarray(object_points, dtype=np.float64).T
            + np.asarray(tvec, dtype=np.float64).reshape(3, 1)
        ).T
        predicted = camera_points[:, :2] / camera_points[:, 2:3]
        observed = _inverse_brown_normalized(image_points, intrinsics)
        scale = np.array([intrinsics["fx"], intrinsics["fy"]], dtype=np.float64)
        return float(np.sqrt(np.mean(((predicted - observed) * scale) ** 2)))

    projected, _ = cv2.projectPoints(
        np.asarray(object_points),
        rvec,
        tvec,
        camera_matrix(intrinsics),
        coefficients if has_distortion else None,
    )
    return float(
        np.sqrt(
            np.mean(
                (projected.reshape(-1, 2) - np.asarray(image_points).reshape(-1, 2))
                ** 2
            )
        )
    )
