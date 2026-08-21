#!/usr/bin/env python3
"""Calculate a pixel-to-2D-plane homography from one chessboard image."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from homography_common import (
    detect_chessboard,
    homography_payload,
    load_intrinsics_json,
    parse_pattern,
    reprojection_error,
    save_payload,
    solve_homography,
    solve_camera_plane_pose,
    undistort_image,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calculate H that maps image pixels to chessboard-plane coordinates"
    )
    parser.add_argument("image", type=Path, help="chessboard image")
    parser.add_argument(
        "--pattern", default="9x6", metavar="C x R",
        help="inner corner count, for example 9x6 (default: 9x6)",
    )
    parser.add_argument(
        "--square-size", type=float, required=True,
        help="physical size of one square; output uses this unit",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("pixel_to_plane_homography.json"),
        help="output JSON path",
    )
    parser.add_argument(
        "--visualization", type=Path,
        help="optional output image with detected corners",
    )
    parser.add_argument(
        "--intrinsics", type=Path,
        default=Path(__file__).resolve().parent.parent
        / "monocular_rgb_calibration" / "intrinsics.json",
        help="camera intrinsics JSON (default: ../monocular_rgb_calibration/intrinsics.json)",
    )
    parser.add_argument(
        "--undistort", action=argparse.BooleanOptionalAction, default=False,
        help="undistort the image before detection (default: false)",
    )
    parser.add_argument(
        "--zero-distortion", action=argparse.BooleanOptionalAction, default=True,
        help="pass zero distortion coefficients to solvePnP (default: true)",
    )
    args = parser.parse_args()

    try:
        pattern = parse_pattern(args.pattern)
        image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"could not read image: {args.image}")
        camera_matrix, dist_coeffs = load_intrinsics_json(
            args.intrinsics, (image.shape[1], image.shape[0])
        )
        working_image = (
            undistort_image(image, camera_matrix, dist_coeffs)
            if args.undistort else image
        )
        corners = detect_chessboard(working_image, pattern)
        if corners is None:
            raise ValueError(
                f"chessboard {args.pattern} was not detected in {args.image}"
            )
        matrix, plane_points, inliers = solve_homography(
            corners, pattern, args.square_size
        )
        errors = reprojection_error(matrix, corners, plane_points)
        payload = homography_payload(
            matrix, pattern, args.square_size,
            (working_image.shape[1], working_image.shape[0]), inliers, errors,
        )
        camera_plane, pnp_rms = solve_camera_plane_pose(
            corners, pattern, args.square_size, camera_matrix,
            np.zeros_like(dist_coeffs)
            if args.zero_distortion or args.undistort else dist_coeffs,
        )
        payload["matrix_camera_plane"] = camera_plane.tolist()
        payload["pnp_reprojection_rms"] = pnp_rms
        payload["undistorted"] = args.undistort
        payload["pnp_zero_distortion"] = args.zero_distortion or args.undistort
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_payload(args.output, payload)
        if args.visualization:
            from homography_common import draw_detection
            if not cv2.imwrite(str(args.visualization), draw_detection(working_image, corners, pattern)):
                raise RuntimeError(f"could not write visualization: {args.visualization}")
    except (OSError, RuntimeError, ValueError, cv2.error) as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Saved homography: {args.output}")
    print("H_pixel_to_plane =")
    for row in matrix:
        print("  " + " ".join(f"{value:.12g}" for value in row))
    print(f"Mean reprojection error: {errors.mean():.4f}")
    print(f"Max reprojection error:  {errors.max():.4f}")
    print("Mapping: [X, Y, W] = H @ [u, v, 1], then X/=W and Y/=W")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
