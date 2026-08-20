#!/usr/bin/env python3
"""Calculate a pixel-to-2D-plane homography from one chessboard image."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from homography_common import (
    detect_chessboard,
    homography_payload,
    parse_pattern,
    reprojection_error,
    save_payload,
    solve_homography,
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
    args = parser.parse_args()

    try:
        pattern = parse_pattern(args.pattern)
        image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"could not read image: {args.image}")
        corners = detect_chessboard(image, pattern)
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
            (image.shape[1], image.shape[0]), inliers, errors,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_payload(args.output, payload)
        if args.visualization:
            from homography_common import draw_detection
            if not cv2.imwrite(str(args.visualization), draw_detection(image, corners, pattern)):
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
