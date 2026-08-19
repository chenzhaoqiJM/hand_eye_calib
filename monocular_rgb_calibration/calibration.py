"""Chessboard detection and pinhole camera calibration helpers."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    valid: bool
    corners: np.ndarray | None
    annotated: np.ndarray
    message: str
    sharpness: float
    coverage: float


def chessboard_object_points(columns: int, rows: int, square_size_mm: float) -> np.ndarray:
    points = np.zeros((columns * rows, 3), dtype=np.float32)
    points[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    points[:, :2] *= float(square_size_mm)
    return points


def detect_chessboard(
    image: np.ndarray,
    columns: int,
    rows: int,
    min_sharpness: float,
    min_coverage: float,
) -> Detection:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    annotated = image.copy()
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, (columns, rows), flags)
    coverage = 0.0
    if found and corners is not None:
        corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )
        x, y, width, height = cv2.boundingRect(corners)
        coverage = float(width * height) / float(gray.shape[0] * gray.shape[1])
        cv2.drawChessboardCorners(annotated, (columns, rows), corners, True)

    if not found or corners is None:
        valid, message = False, "INVALID: chessboard not found"
    elif sharpness < min_sharpness:
        valid, message = False, f"INVALID: blurred ({sharpness:.1f} < {min_sharpness:.1f})"
    elif coverage < min_coverage:
        valid, message = False, f"INVALID: board too small ({coverage:.1%} < {min_coverage:.1%})"
    else:
        valid, message = True, "VALID: ready to capture"

    color = (40, 210, 40) if valid else (30, 30, 230)
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 70), (20, 20, 20), -1)
    cv2.putText(annotated, message, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    cv2.putText(
        annotated,
        f"sharpness={sharpness:.1f}  coverage={coverage:.1%}",
        (16, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (240, 240, 240),
        1,
    )
    return Detection(valid, corners, annotated, message, sharpness, coverage)


def calibrate_camera(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    image_size: tuple[int, int],
) -> tuple[float, np.ndarray, np.ndarray, list[float]]:
    rms, matrix, coefficients, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    errors: list[float] = []
    for points_3d, points_2d, rvec, tvec in zip(
        object_points, image_points, rvecs, tvecs, strict=True
    ):
        projected, _ = cv2.projectPoints(points_3d, rvec, tvec, matrix, coefficients)
        error = cv2.norm(points_2d, projected, cv2.NORM_L2) / len(projected)
        errors.append(float(error))
    return float(rms), matrix, coefficients.reshape(-1), errors
