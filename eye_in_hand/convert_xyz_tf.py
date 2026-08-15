#!/usr/bin/env python3
"""Convert a YOLOE grasp point from camera coordinates to robot base coordinates.

Transform convention:

    p_base = T_base_flange @ T_flange_camera @ p_camera

The input point and both transform translations are expected to be in metres.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_HAND_EYE = HERE / "T_flange_camera.json"
DEFAULT_GRASP_RESULT = (
    HERE.parent / "yoloe_grasp_front" / "outputs" / "d405_grasp" / "grasp_result.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert grasp point camera -> flange -> robot base using ROS 2 TF."
    )
    parser.add_argument("--grasp-result", type=Path, default=DEFAULT_GRASP_RESULT)
    parser.add_argument("--hand-eye", type=Path, default=DEFAULT_HAND_EYE)
    parser.add_argument("--base-frame", default="base_footprint_link")
    parser.add_argument("--flange-frame", default="Right_Arm_Link8")
    parser.add_argument("--ros-domain-id", type=int, default=25)
    parser.add_argument("--timeout", type=float, default=5.0)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def validate_transform(value, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 matrix, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(f"{name} has an invalid homogeneous last row")
    return matrix


def load_camera_point(path: Path) -> np.ndarray:
    result = load_json(path)
    try:
        point = np.asarray(result["grasp"]["point_camera_m"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{path} does not contain a valid grasp.point_camera_m"
        ) from exc
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError("grasp.point_camera_m must contain three finite values")
    return np.append(point, 1.0)


def get_T_base_flange(
    base_frame: str,
    flange_frame: str,
    timeout_seconds: float,
) -> tuple[np.ndarray, float | None]:
    try:
        import rclpy
        from rclpy.duration import Duration
        from rclpy.time import Time
        from scipy.spatial.transform import Rotation
        from tf2_ros import Buffer, TransformListener
    except ImportError as exc:
        raise RuntimeError(
            "ROS 2 Python packages are unavailable. Run this script in a terminal "
            "where the ROS 2 environment has been loaded."
        ) from exc

    rclpy.init()
    node = rclpy.create_node("convert_xyz_tf")
    tf_buffer = Buffer()
    listener = TransformListener(tf_buffer, node)

    try:
        deadline = time.monotonic() + timeout_seconds
        transform = None
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            try:
                transform = tf_buffer.lookup_transform(base_frame, flange_frame, Time())
                break
            except Exception as exc:  # tf2_ros exposes several lookup exception types.
                last_error = exc

        if transform is None:
            raise RuntimeError(
                f"No TF {base_frame} <- {flange_frame} received within "
                f"{timeout_seconds:g} s"
            ) from last_error

        translation = transform.transform.translation
        quaternion = transform.transform.rotation
        quat_xyzw = np.asarray(
            [quaternion.x, quaternion.y, quaternion.z, quaternion.w],
            dtype=np.float64,
        )
        quat_norm = float(np.linalg.norm(quat_xyzw))
        if not np.isfinite(quat_norm) or quat_norm < 1e-12:
            raise ValueError("TF contains an invalid quaternion")
        quat_xyzw /= quat_norm

        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
        matrix[:3, 3] = [translation.x, translation.y, translation.z]

        stamp = transform.header.stamp
        stamp_seconds = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        return matrix, stamp_seconds if stamp_seconds > 0 else None
    finally:
        del listener
        node.destroy_node()
        rclpy.shutdown()


def format_xyz(point: np.ndarray) -> str:
    return "[" + ", ".join(f"{value:.6f}" for value in point[:3]) + "] m"


def main() -> int:
    args = parse_args()
    if not 0 <= args.ros_domain_id <= 232:
        raise SystemExit("--ros-domain-id must be between 0 and 232")
    if not np.isfinite(args.timeout) or args.timeout <= 0:
        raise SystemExit("--timeout must be positive")

    # ROS_DOMAIN_ID must be set before importing/initializing rclpy.
    os.environ["ROS_DOMAIN_ID"] = str(args.ros_domain_id)

    try:
        p_camera = load_camera_point(args.grasp_result)
        hand_eye_data = load_json(args.hand_eye)
        T_flange_camera = validate_transform(
            hand_eye_data["matrix_4x4"], "T_flange_camera"
        )
        T_base_flange, tf_stamp_seconds = get_T_base_flange(
            args.base_frame,
            args.flange_frame,
            args.timeout,
        )
    except (FileNotFoundError, KeyError, OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    p_flange = T_flange_camera @ p_camera
    p_base = T_base_flange @ p_flange

    print(f"TF: {args.base_frame} <- {args.flange_frame}")
    if tf_stamp_seconds is not None:
        print(f"TF timestamp: {tf_stamp_seconds:.9f} s")
    else:
        print("TF timestamp: static/latest")
    print("T_base_flange:")
    print(np.array2string(T_base_flange, precision=7, suppress_small=True))
    print(f"Point in camera: {format_xyz(p_camera)}")
    print(f"Point in flange: {format_xyz(p_flange)}")
    print(f"Point in base:   {format_xyz(p_base)}")

    if not bool(hand_eye_data.get("validated", False)):
        print("WARNING: hand-eye calibration is marked unvalidated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
