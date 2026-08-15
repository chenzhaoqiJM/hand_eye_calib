#!/usr/bin/env python3
"""Collect hand-eye images and matching flange poses from ROS 2 TF.

Press Enter once the robot is stationary.  The latest camera image and the
latest ``base_footprint_link <- Right_Arm_Link8`` transform are saved using the
same on-disk format as calibrate.py/calibrate_from_data.py.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

from calibrate import (
    DEFAULT_MIN_SAMPLES,
    DEFAULT_TAG_SIZE_MM,
    FrameStream,
    HERE,
    PreviewServer,
    append_pose_text,
    local_preview_urls,
    save_intrinsics,
    write_image,
)
from calibrate_from_data import solve_from_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ROS 2 TF hand-eye collection: D405 images + flange TF poses"
    )
    parser.add_argument("--tag-size-mm", type=float, default=DEFAULT_TAG_SIZE_MM)
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES)
    parser.add_argument("--robot-id", type=int, choices=[0, 1], default=1)
    parser.add_argument("--base-frame", default="Body_Link5")
    parser.add_argument("--flange-frame", default="Right_Arm_Link8")
    parser.add_argument("--ros-domain-id", type=int, default=25)
    parser.add_argument(
        "--max-tf-age",
        type=float,
        default=2.5,
        help="Reject a non-static TF older than this many seconds (default: 2.5)",
    )
    parser.add_argument("--tf-wait-timeout", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--preview-host", default="0.0.0.0")
    parser.add_argument("--preview-port", type=int, default=8080)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--no-solve", action="store_true")
    args = parser.parse_args()

    if args.min_samples < 8:
        parser.error("--min-samples must be at least 8")
    if not np.isfinite(args.tag_size_mm) or args.tag_size_mm <= 0:
        parser.error("--tag-size-mm must be positive")
    if not 0 <= args.ros_domain_id <= 232:
        parser.error("--ros-domain-id must be between 0 and 232")
    if not np.isfinite(args.max_tf_age) or args.max_tf_age <= 0:
        parser.error("--max-tf-age must be positive")
    if not np.isfinite(args.tf_wait_timeout) or args.tf_wait_timeout <= 0:
        parser.error("--tf-wait-timeout must be positive")
    if not 1 <= args.preview_port <= 65535:
        parser.error("--preview-port must be between 1 and 65535")
    return args


def transform_to_sample(transform_stamped) -> tuple[np.ndarray, dict]:
    """Convert geometry_msgs/TransformStamped to XYZ + XYZ Euler radians."""
    translation = transform_stamped.transform.translation
    quaternion = transform_stamped.transform.rotation
    quat_xyzw = np.asarray(
        [quaternion.x, quaternion.y, quaternion.z, quaternion.w], dtype=np.float64
    )
    norm = float(np.linalg.norm(quat_xyzw))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("TF contains an invalid zero/non-finite quaternion")
    quat_xyzw /= norm
    rotation = Rotation.from_quat(quat_xyzw)
    rpy = rotation.as_euler("xyz")
    pose = np.asarray(
        [translation.x, translation.y, translation.z, *rpy], dtype=np.float64
    )
    if not np.all(np.isfinite(pose)):
        raise ValueError("TF contains non-finite translation or rotation values")
    stamp = transform_stamped.header.stamp
    metadata = {
        "tf_stamp": {"sec": int(stamp.sec), "nanosec": int(stamp.nanosec)},
        "tf_stamp_seconds": float(stamp.sec) + float(stamp.nanosec) * 1e-9,
        "quaternion_xyzw": quat_xyzw.tolist(),
        "matrix_4x4": np.block(
            [
                [rotation.as_matrix(), pose[:3, None]],
                [np.asarray([[0.0, 0.0, 0.0]]), np.asarray([[1.0]])],
            ]
        ).tolist(),
    }
    return pose, metadata


def save_tf_pose(
    save_dir: Path,
    index: int,
    pose: np.ndarray,
    metadata: dict,
    base_frame: str,
    flange_frame: str,
) -> None:
    payload = {
        "x": float(pose[0]),
        "y": float(pose[1]),
        "z": float(pose[2]),
        "rx": float(pose[3]),
        "ry": float(pose[4]),
        "rz": float(pose[5]),
        "position_unit": "m",
        "angle_unit": "rad",
        "pose_convention": "XYZ Euler: R = Rz(rz) @ Ry(ry) @ Rx(rx)",
        "target_frame": base_frame,
        "source_frame": flange_frame,
        "captured_at": datetime.now().astimezone().isoformat(timespec="microseconds"),
        **metadata,
    }
    (save_dir / f"pose_{index:03d}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def write_session(camera, save_dir: Path, args: argparse.Namespace) -> None:
    payload = {
        "robot_id": args.robot_id,
        "camera_mount": "eye_in_hand",
        "output_transform": "T_flange_camera",
        "tag_family": "DICT_APRILTAG_36h11",
        "tag_size_mm": float(args.tag_size_mm),
        "pose_source": "ros2_tf",
        "pose_position_unit": "m",
        "pose_angle_unit": "rad",
        "tf_target_frame": args.base_frame,
        "tf_source_frame": args.flange_frame,
        "ros_domain_id": args.ros_domain_id,
        "camera_device": camera.device_info,
        "camera_stream": camera.stream_config,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    (save_dir / "session.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def wait_for_tf(buffer, node, args: argparse.Namespace):
    from rclpy.time import Time

    deadline = time.monotonic() + args.tf_wait_timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            return buffer.lookup_transform(args.base_frame, args.flange_frame, Time())
        except Exception as exc:  # tf2_ros has several lookup exception classes
            last_error = exc
            time.sleep(0.05)
    raise RuntimeError(
        f"No TF {args.base_frame} <- {args.flange_frame} received within "
        f"{args.tf_wait_timeout:g} s. Check ROS_DOMAIN_ID, frame names, network, "
        "and Windows Firewall."
    ) from last_error


def tf_age_seconds(node, transform_stamped) -> float | None:
    stamp = transform_stamped.header.stamp
    stamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    if stamp_ns == 0:  # A static transform commonly uses the zero timestamp.
        return None
    return (node.get_clock().now().nanoseconds - stamp_ns) * 1e-9


def collect(args: argparse.Namespace) -> Path:
    # Domain selection must happen before importing/initializing rclpy.
    os.environ["ROS_DOMAIN_ID"] = str(args.ros_domain_id)
    try:
        import rclpy
        from tf2_ros import Buffer, TransformListener
    except ImportError as exc:
        raise RuntimeError(
            "ROS 2 Python packages were not found. Run this script in a terminal "
            "where your ROS 2 installation has been sourced."
        ) from exc

    from camera import Camera

    save_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else HERE / "data" / datetime.now().strftime("%Y-%m-%d_%H%M%S_ros2_tf")
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    rclpy.init(args=None)
    node = rclpy.create_node("hand_eye_tf_collector")
    tf_buffer = Buffer()
    tf_listener = TransformListener(tf_buffer, node, spin_thread=True)
    camera = Camera(width=args.width, height=args.height, fps=args.fps)
    camera_started = False
    frame_stream: FrameStream | None = None
    preview_server: PreviewServer | None = None
    try:
        print(
            f"Waiting for TF: {args.base_frame} <- {args.flange_frame} "
            f"(ROS_DOMAIN_ID={args.ros_domain_id})"
        )
        first_tf = wait_for_tf(tf_buffer, node, args)
        first_pose, _ = transform_to_sample(first_tf)
        print(
            "TF ready. XYZ="
            f"({first_pose[0]:.4f}, {first_pose[1]:.4f}, {first_pose[2]:.4f}) m, "
            "RPY="
            f"({np.degrees(first_pose[3]):.2f}, {np.degrees(first_pose[4]):.2f}, "
            f"{np.degrees(first_pose[5]):.2f}) deg"
        )

        camera.start()
        camera_started = True
        save_intrinsics(camera, save_dir)
        write_session(camera, save_dir, args)
        if not args.no_preview:
            frame_stream = FrameStream(camera)
            frame_stream.start()
            preview_server = PreviewServer(
                frame_stream, args.preview_host, args.preview_port
            )
            preview_server.start()

        print(f"Camera ready. Saving data to: {save_dir}")
        if preview_server is not None:
            print("Browser preview:")
            for url in local_preview_urls(args.preview_host, preview_server.port):
                print(f"  {url}")
        print("Stop the robot at varied poses and keep exactly one AprilTag visible.")
        print("Each Enter saves the current image and latest TF together.")

        index = 0
        while True:
            command = input(f"\n[{index:03d}] Enter=capture, q=finish: ").strip()
            if command.lower() == "q":
                if index < args.min_samples:
                    print(f"Need at least {args.min_samples} samples, currently {index}.")
                    continue
                break
            if command:
                print("Use Enter to capture or q to finish.")
                continue

            # Perform both reads consecutively at the keypress. The camera stream
            # gives the latest complete frame, while Time() requests the newest TF.
            image = frame_stream.snapshot() if frame_stream else camera.capture()
            transform = wait_for_tf(tf_buffer, node, args)
            age = tf_age_seconds(node, transform)
            if age is not None and (age < -0.1 or age > args.max_tf_age):
                print(
                    f"  Not saved: latest TF age is {age:.3f} s; expected 0 to "
                    f"{args.max_tf_age:g} s. Check clock synchronization/publisher."
                )
                continue

            pose, metadata = transform_to_sample(transform)
            image_path = save_dir / f"frame_{index:03d}.png"
            pose_path = save_dir / f"pose_{index:03d}.json"
            write_image(image_path, image)
            try:
                save_tf_pose(
                    save_dir,
                    index,
                    pose,
                    metadata,
                    args.base_frame,
                    args.flange_frame,
                )
                append_pose_text(save_dir, index, pose)
            except BaseException:
                image_path.unlink(missing_ok=True)
                pose_path.unlink(missing_ok=True)
                raise
            age_text = "static" if age is None else f"age={age:.3f} s"
            print(
                f"  Saved frame_{index:03d}.png + pose_{index:03d}.json ({age_text})\n"
                f"  XYZ=({pose[0]:.4f}, {pose[1]:.4f}, {pose[2]:.4f}) m, "
                f"RPY=({np.degrees(pose[3]):.2f}, {np.degrees(pose[4]):.2f}, "
                f"{np.degrees(pose[5]):.2f}) deg"
            )
            index += 1
    finally:
        if preview_server is not None:
            preview_server.stop()
        if frame_stream is not None:
            frame_stream.stop()
        if camera_started:
            camera.stop()
        del tf_listener
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return save_dir


def main() -> int:
    args = parse_args()
    try:
        data_dir = collect(args)
    except (KeyboardInterrupt, EOFError):
        print("\nCollection interrupted.")
        return 130
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"\nCollection complete: {data_dir}")
    if not args.no_solve:
        solve_from_data(data_dir, args.tag_size_mm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
