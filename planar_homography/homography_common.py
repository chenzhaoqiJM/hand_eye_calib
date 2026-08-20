#!/usr/bin/env python3
"""Shared chessboard detection and pixel-to-plane homography helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def parse_pattern(value: str) -> tuple[int, int]:
    try:
        columns, rows = (int(item) for item in value.lower().split("x", 1))
    except (ValueError, TypeError) as exc:
        raise ValueError("pattern must look like CxR, for example 9x6") from exc
    if columns < 2 or rows < 2:
        raise ValueError("pattern dimensions must both be at least 2")
    return columns, rows


def detect_chessboard(
    image: np.ndarray,
    pattern: tuple[int, int],
) -> np.ndarray | None:
    """Return refined inner corners in row-major OpenCV chessboard order."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern, flags)
    if not found or corners is None:
        return None
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return refined.reshape(-1, 2).astype(np.float64)


def draw_detection(
    image: np.ndarray,
    corners: np.ndarray | None,
    pattern: tuple[int, int],
) -> np.ndarray:
    output = image.copy()
    if corners is not None:
        # OpenCV 4.14 requires float32 for drawChessboardCorners input points.
        drawable_corners = np.ascontiguousarray(corners, dtype=np.float32)
        cv2.drawChessboardCorners(
            output, pattern, drawable_corners.reshape(-1, 1, 2), True
        )
    return output


def load_intrinsics(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    try:
        matrix = np.asarray(data["camera_matrix"], dtype=np.float64)
        coeffs = np.asarray(data["dist_coeffs"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "intrinsics JSON must contain camera_matrix and dist_coeffs"
        ) from exc
    if matrix.shape != (3, 3) or coeffs.size < 4:
        raise ValueError("invalid camera_matrix or dist_coeffs shape")
    return matrix, coeffs.reshape(-1, 1)


def solve_homography(
    corners: np.ndarray,
    pattern: tuple[int, int],
    square_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if corners.shape != (pattern[0] * pattern[1], 2):
        raise ValueError("detected corner count does not match pattern")
    if not np.isfinite(square_size) or square_size <= 0:
        raise ValueError("square_size must be a positive finite number")

    columns, rows = pattern
    plane_points = np.array(
        [(column * square_size, row * square_size)
         for row in range(rows) for column in range(columns)],
        dtype=np.float64,
    )
    matrix, mask = cv2.findHomography(corners, plane_points, cv2.RANSAC, 3.0)
    if matrix is None or mask is None:
        raise RuntimeError("could not calculate homography")
    inliers = mask.reshape(-1).astype(bool)
    if int(inliers.sum()) < 4:
        raise RuntimeError("homography has fewer than four inlier points")
    matrix /= matrix[2, 2]
    return matrix, plane_points, inliers


def reprojection_error(
    matrix: np.ndarray,
    image_points: np.ndarray,
    plane_points: np.ndarray,
) -> np.ndarray:
    projected = cv2.perspectiveTransform(
        image_points.reshape(-1, 1, 2).astype(np.float64), matrix
    ).reshape(-1, 2)
    return np.linalg.norm(projected - plane_points, axis=1)


def homography_payload(
    matrix: np.ndarray,
    pattern: tuple[int, int],
    square_size: float,
    image_size: tuple[int, int],
    inliers: np.ndarray,
    errors: np.ndarray,
) -> dict[str, Any]:
    return {
        "type": "pixel_to_chessboard_plane_homography",
        "matrix_pixel_to_plane": matrix.tolist(),
        "mapping": "[x, y, w] = H @ [u, v, 1]; X=x/w, Y=y/w",
        "plane_coordinate_unit": "same_as_square_size",
        "plane_origin": "first_detected_inner_corner",
        "plane_axes": "X follows columns, Y follows rows",
        "pattern_inner_corners": {"columns": pattern[0], "rows": pattern[1]},
        "square_size": float(square_size),
        "image_size": {"width": image_size[0], "height": image_size[1]},
        "inlier_count": int(inliers.sum()),
        "corner_count": int(inliers.size),
        "max_reprojection_error": float(errors.max()),
        "mean_reprojection_error": float(errors.mean()),
    }


def save_payload(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
