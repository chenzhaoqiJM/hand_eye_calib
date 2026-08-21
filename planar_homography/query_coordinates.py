#!/usr/bin/env python3
"""Interactively convert image pixels to plane and camera coordinates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_calibration(path: Path) -> dict[str, Any]:
    """Load and validate the homography output JSON."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取标定文件 {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("标定文件顶层必须是 JSON 对象")

    try:
        homography = np.asarray(data["matrix_pixel_to_plane"], dtype=np.float64)
        camera_plane = np.asarray(data["matrix_camera_plane"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "标定文件必须包含 matrix_pixel_to_plane 和 matrix_camera_plane"
        ) from exc
    if homography.shape != (3, 3) or not np.all(np.isfinite(homography)):
        raise ValueError("matrix_pixel_to_plane 必须是有限的 3x3 矩阵")
    if camera_plane.shape != (4, 4) or not np.all(np.isfinite(camera_plane)):
        raise ValueError("matrix_camera_plane 必须是有限的 4x4 矩阵")

    data["matrix_pixel_to_plane"] = homography
    data["matrix_camera_plane"] = camera_plane
    return data


def convert_pixel(
    pixel: tuple[float, float],
    homography: np.ndarray,
    camera_plane: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (plane XYZ, camera XYZ) for an image pixel.

    ``matrix_camera_plane`` is T_camera_plane, so it maps a point from
    the calibration-board frame into the camera frame.
    """
    pixel_homogeneous = np.array([pixel[0], pixel[1], 1.0], dtype=np.float64)
    plane_homogeneous = homography @ pixel_homogeneous
    scale = float(plane_homogeneous[2])
    if abs(scale) < 1e-12:
        raise ValueError("该像素对应的单应性齐次尺度接近 0，无法计算平面坐标")

    plane_xy = plane_homogeneous[:2] / scale
    plane_homogeneous = np.array(
        [plane_xy[0], plane_xy[1], 0.0, 1.0], dtype=np.float64
    )
    camera_homogeneous = camera_plane @ plane_homogeneous
    camera_scale = float(camera_homogeneous[3])
    if abs(camera_scale) < 1e-12:
        raise ValueError("相机坐标齐次尺度接近 0，无法计算相机坐标")
    camera_xyz = camera_homogeneous[:3] / camera_scale
    return np.array([plane_xy[0], plane_xy[1], 0.0]), camera_xyz


def parse_pixel(value: str) -> tuple[float, float]:
    """Parse ``u v`` or ``u,v`` entered by the user."""
    fields = value.replace(",", " ").split()
    if len(fields) != 2:
        raise ValueError("请输入两个数字，格式为: u v")
    try:
        pixel = (float(fields[0]), float(fields[1]))
    except ValueError as exc:
        raise ValueError("像素坐标必须是数字") from exc
    if not np.all(np.isfinite(pixel)):
        raise ValueError("像素坐标必须是有限数字")
    return pixel


def print_result(
    pixel: tuple[float, float],
    plane_xyz: np.ndarray,
    camera_xyz: np.ndarray,
    unit: str,
) -> None:
    """Print one coordinate conversion in a readable format."""
    print(f"\n像素坐标:       (u={pixel[0]:.6f}, v={pixel[1]:.6f})")
    print(
        "标定板坐标系:   "
        f"(X={plane_xyz[0]:.6f}, Y={plane_xyz[1]:.6f}, Z={plane_xyz[2]:.6f}) {unit}"
    )
    print(
        "相机坐标系:     "
        f"(X={camera_xyz[0]:.6f}, Y={camera_xyz[1]:.6f}, Z={camera_xyz[2]:.6f}) {unit}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="将像素坐标转换为标定板坐标系和相机坐标系坐标"
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path(__file__).resolve().parent / "pixel_to_plane_homography.json",
        help="标定输出 JSON（默认: pixel_to_plane_homography.json）",
    )
    parser.add_argument(
        "--pixel",
        nargs=2,
        type=float,
        metavar=("U", "V"),
        help="直接转换一个像素坐标；不指定时进入交互模式",
    )
    args = parser.parse_args()

    try:
        data = load_calibration(args.calibration)
    except ValueError as exc:
        parser.error(str(exc))

    homography = data["matrix_pixel_to_plane"]
    camera_plane = data["matrix_camera_plane"]
    unit = str(data.get("plane_coordinate_unit", "same_as_square_size"))
    if unit == "same_as_square_size":
        unit = f"square_size units (square_size={data.get('square_size', '?')})"
    image_size = data.get("image_size")
    undistorted = bool(data.get("undistorted", False))

    print(f"已加载标定文件: {args.calibration}")
    if isinstance(image_size, dict):
        print(
            f"适用图像尺寸: {image_size.get('width', '?')}x"
            f"{image_size.get('height', '?')}"
        )
    print(f"坐标单位: {unit}")
    if undistorted:
        print("注意: 此标定基于去畸变图像，输入的 (u, v) 也必须是去畸变像素坐标。")
    print("输入格式: u v（也支持 u,v）；输入 q 或 quit 退出。")

    def convert_and_print(pixel: tuple[float, float]) -> None:
        if isinstance(image_size, dict):
            width = image_size.get("width")
            height = image_size.get("height")
            if isinstance(width, (int, float)) and isinstance(height, (int, float)):
                if not (0 <= pixel[0] < width and 0 <= pixel[1] < height):
                    print("警告: 该像素超出标定图像范围，结果属于单应性外推。")
        plane_xyz, camera_xyz = convert_pixel(pixel, homography, camera_plane)
        print_result(pixel, plane_xyz, camera_xyz, unit)

    if args.pixel is not None:
        pixel = (args.pixel[0], args.pixel[1])
        if not np.all(np.isfinite(pixel)):
            parser.error("--pixel 的坐标必须是有限数字")
        try:
            convert_and_print(pixel)
        except ValueError as exc:
            parser.error(str(exc))
        return 0

    while True:
        try:
            value = input("\n请输入像素坐标 (u v): ").strip()
        except EOFError:
            print()
            return 0
        if value.lower() in {"q", "quit", "exit"}:
            return 0
        try:
            convert_and_print(parse_pixel(value))
        except ValueError as exc:
            print(f"输入错误: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())