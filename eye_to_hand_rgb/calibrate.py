#!/usr/bin/env python3
"""Collect V4L2 RGB images and manual flange poses for eye-to-hand calibration."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import cv2
import numpy as np

from calibrate_from_data import (
    chessboard_object_points,
    create_tag_detector,
    detect_target_pose,
    normalize_pose_units,
    solve_from_data,
    tag_points,
)
from camera import Camera
from camera_model import validate_intrinsics
from preview import FrameStream, PreviewServer, preview_urls


HERE = Path(__file__).resolve().parent


def load_intrinsics(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("intrinsics JSON must contain an object")
    validate_intrinsics(value)
    return value


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def write_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"Could not encode image: {path}")
    encoded.tofile(str(path))


def assess_target(
    target: dict | None,
    rejection: dict | None,
    target_type: str,
    min_board_coverage: float,
    max_reprojection_px: float,
) -> tuple[bool, str]:
    if target is None:
        return False, "unknown" if rejection is None else str(rejection["reason"])
    if target["reprojection_rms_px"] > max_reprojection_px:
        return False, "reprojection_high"
    if target_type == "chessboard" and target["coverage"] < min_board_coverage:
        return False, "board_too_small"
    return True, "ok"


def draw_quality_overlay(
    image: np.ndarray,
    target: dict | None,
    rejection: dict | None,
    target_type: str,
    chessboard_columns: int,
    chessboard_rows: int,
    min_board_coverage: float,
    max_reprojection_px: float,
) -> np.ndarray:
    """Draw target detection and quality information on the live preview."""
    overlay = image.copy()
    target_type = target_type.lower()
    if target is not None:
        image_points = np.asarray(target["image_points"], dtype=np.float32)
        if target_type == "chessboard":
            cv2.drawChessboardCorners(
                overlay,
                (chessboard_columns, chessboard_rows),
                image_points.reshape(-1, 1, 2),
                True,
            )
        else:
            cv2.polylines(
                overlay,
                [image_points.reshape(-1, 1, 2).astype(np.int32)],
                True,
                (255, 180, 0),
                2,
            )
        valid, reason = assess_target(
            target,
            rejection,
            target_type,
            min_board_coverage,
            max_reprojection_px,
        )
        color = (70, 220, 70) if valid else (0, 210, 255)
        lines = [
            f"{target_type.upper()}: DETECTED",
            f"VALID: {'YES' if valid else 'NO'} ({reason})",
            f"RMS: {target['reprojection_rms_px']:.3f} px / max {max_reprojection_px:.3f}",
        ]
        if target_type == "chessboard":
            lines.append(
                f"COVERAGE: {target['coverage']:.1%} / min {min_board_coverage:.1%}"
            )
    else:
        reason = "unknown" if rejection is None else str(rejection["reason"])
        color = (0, 0, 255)
        lines = [
            f"{target_type.upper()}: NOT FOUND",
            f"VALID: NO ({reason})",
            "RMS: -",
            "COVERAGE: -" if target_type == "chessboard" else "",
        ]

    lines = [line for line in lines if line]
    panel_height = 18 + 27 * len(lines)
    panel_width = min(760, overlay.shape[1] - 20)
    panel = overlay.copy()
    cv2.rectangle(panel, (10, 10), (10 + panel_width, 10 + panel_height), (15, 20, 25), -1)
    cv2.addWeighted(panel, 0.78, overlay, 0.22, 0.0, overlay)
    for offset, line in enumerate(lines):
        cv2.putText(
            overlay,
            line,
            (22, 35 + offset * 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            color,
            2,
            cv2.LINE_AA,
        )
    return overlay


def prompt_pose(index: int, position_unit: str, angle_unit: str) -> np.ndarray | None:
    while True:
        line = input(
            f"[{index:03d}] T_base_flange x y z rx ry rz "
            f"({position_unit}, {angle_unit}); q=finish, r=retake: "
        ).strip()
        if line.lower() == "q":
            return None
        if line.lower() == "r":
            raise LookupError("retake")
        try:
            parts = line.replace(",", " ").split()
            if len(parts) != 6:
                raise ValueError("enter exactly 6 numbers")
            return normalize_pose_units(parts, position_unit, angle_unit)
        except ValueError as exc:
            print(f"  Invalid pose: {exc}")


def save_pose(data_dir: Path, index: int, pose: np.ndarray) -> None:
    keys = ("x", "y", "z", "rx", "ry", "rz")
    payload = dict(zip(keys, map(float, pose), strict=True))
    payload.update({"position_unit": "m", "angle_unit": "rad"})
    write_json(data_dir / f"pose_{index:03d}.json", payload)


def collect(args: argparse.Namespace) -> Path:
    intrinsics_source = (
        None if args.intrinsics is None else Path(args.intrinsics).resolve()
    )
    intrinsics = None if intrinsics_source is None else load_intrinsics(intrinsics_source)
    data_dir = Path(args.output_dir).resolve() if args.output_dir else (
        HERE / "data" / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    )
    data_dir.mkdir(parents=True, exist_ok=True)
    if args.camera_backend == "realsense":
        from camera_realsense import RealSenseCamera

        camera = RealSenseCamera(
            width=args.width,
            height=args.height,
            fps=args.fps,
            serial_number=args.serial_number,
        )
    else:
        camera = Camera(args.device, args.width, args.height, args.fps, args.pixel_format)
    if args.target_type == "chessboard":
        target_detector = None
        target_points = chessboard_object_points(
            args.chessboard_columns, args.chessboard_rows, args.square_size_mm
        )
    else:
        target_detector = create_tag_detector()
        target_points = tag_points(args.tag_size_mm)
    frame_stream: FrameStream | None = None
    preview_server: PreviewServer | None = None
    try:
        camera.start()
        width, height = camera.resolution
        if intrinsics is None:
            intrinsics = camera.intrinsics_payload
        validate_intrinsics(intrinsics, width, height)
        write_json(data_dir / "intrinsics.json", intrinsics)
        write_json(data_dir / "session.json", {
            "robot_id": args.robot_id,
            "calibration_type": "eye_to_hand",
            "camera_mount": "fixed_in_workspace",
            "target_mount": "rigid_on_flange",
            "robot_pose": "T_base_flange",
            "output_transform": "T_base_camera",
            "target_type": args.target_type,
            "tag_family": "DICT_APRILTAG_36h11" if args.target_type == "apriltag" else None,
            "tag_size_mm": args.tag_size_mm if args.target_type == "apriltag" else None,
            "chessboard_columns": args.chessboard_columns if args.target_type == "chessboard" else None,
            "chessboard_rows": args.chessboard_rows if args.target_type == "chessboard" else None,
            "square_size_mm": args.square_size_mm if args.target_type == "chessboard" else None,
            "camera_device": camera.device_info,
            "camera_stream": camera.stream_config,
            "camera_backend": args.camera_backend,
            "intrinsics_source": (
                str(intrinsics_source)
                if intrinsics_source is not None else "realsense_device"
            ),
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        })
        def process_preview(frame: np.ndarray) -> np.ndarray:
            try:
                target, rejection = detect_target_pose(
                    frame,
                    target_detector,
                    intrinsics,
                    target_points,
                    args.target_type,
                    args.chessboard_columns,
                    args.chessboard_rows,
                )
                return draw_quality_overlay(
                    frame,
                    target,
                    rejection,
                    args.target_type,
                    args.chessboard_columns,
                    args.chessboard_rows,
                    args.min_board_coverage,
                    args.max_reprojection_px,
                )
            except (ValueError, cv2.error) as exc:
                return draw_quality_overlay(
                    frame,
                    None,
                    {"reason": f"processing_error: {exc}"},
                    args.target_type,
                    args.chessboard_columns,
                    args.chessboard_rows,
                    args.min_board_coverage,
                    args.max_reprojection_px,
                )

        frame_stream = FrameStream(camera, process_preview)
        frame_stream.start()
        if not args.no_preview:
            preview_server = PreviewServer(frame_stream, args.preview_host, args.preview_port)
            preview_server.start()
        print(f"Camera ready: {camera.stream_config}")
        print(f"Saving data to: {data_dir}")
        if preview_server is not None:
            print("Browser preview:")
            for url in preview_urls(args.preview_host, preview_server.port):
                print(f"  {url}")
        target_name = "AprilTag" if args.target_type == "apriltag" else "Chessboard"
        print(f"{target_name} must be rigidly attached to the flange and visible to the fixed camera.")
        print("Use varied flange positions and rotations; stop the robot before each capture.")

        index = 0
        while True:
            command = input(f"\n[{index:03d}] Enter=capture, q=finish: ").strip().lower()
            if command == "q":
                if index < args.min_samples:
                    print(f"Need at least {args.min_samples} samples, currently {index}.")
                    continue
                break
            if command:
                print("Use Enter to capture or q to finish.")
                continue
            image = frame_stream.snapshot()
            target, rejection = detect_target_pose(
                image,
                target_detector,
                intrinsics,
                target_points,
                args.target_type,
                args.chessboard_columns,
                args.chessboard_rows,
            )
            if target is None:
                print(f"  Capture rejected: {rejection['reason']}")
                continue
            valid, reason = assess_target(
                target,
                rejection,
                args.target_type,
                args.min_board_coverage,
                args.max_reprojection_px,
            )
            if not valid:
                details = f"reason={reason}"
                if target is not None:
                    details += (
                        f", PnP RMS {target['reprojection_rms_px']:.3f}px"
                        f", coverage {target.get('coverage', 0.0):.1%}"
                    )
                print(
                    f"  Capture rejected: {details}"
                )
                continue
            if args.target_type == "chessboard":
                print(
                    f"  Chessboard OK: {target['corner_count']} corners, "
                    f"PnP RMS {target['reprojection_rms_px']:.3f}px, "
                    f"coverage {target['coverage']:.1%}"
                )
            else:
                print(
                    f"  AprilTag OK: id={target['tag_id']}, "
                    f"PnP RMS {target['reprojection_rms_px']:.3f}px"
                )
            image_path = data_dir / f"frame_{index:03d}.png"
            write_image(image_path, image)
            print(f"  Saved {image_path.name}")
            try:
                pose = prompt_pose(index, args.position_unit, args.angle_unit)
            except LookupError:
                image_path.unlink(missing_ok=True)
                print("  Retake requested; image removed.")
                continue
            if pose is None:
                image_path.unlink(missing_ok=True)
                if index < args.min_samples:
                    print(f"Need at least {args.min_samples} samples, currently {index}.")
                    continue
                break
            save_pose(data_dir, index, pose)
            print(f"  Saved pose_{index:03d}.json")
            index += 1
    finally:
        if preview_server is not None:
            preview_server.stop()
        if frame_stream is not None:
            frame_stream.stop()
        camera.stop()
    return data_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="RGB eye-to-hand collection and T_base_camera calibration"
    )
    parser.add_argument("--camera-backend", choices=["v4l2", "realsense"], default="v4l2",
                        help="camera backend; default: v4l2")
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--serial-number", default=None,
                        help="RealSense serial number when --camera-backend=realsense")
    parser.add_argument("--intrinsics", default=None,
                        help="RGB camera intrinsics JSON; required for V4L2, optional for RealSense")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--pixel-format", default="MJPG", help="V4L2 FourCC, e.g. MJPG/YUYV")
    parser.add_argument("--target-type", choices=["apriltag", "chessboard"], default="apriltag")
    parser.add_argument("--tag-size-mm", type=float, default=50.0)
    parser.add_argument("--chessboard-columns", type=int, default=7,
                        help="number of chessboard inner corners along X")
    parser.add_argument("--chessboard-rows", type=int, default=6,
                        help="number of chessboard inner corners along Y")
    parser.add_argument("--square-size-mm", type=float, default=40.0)
    parser.add_argument("--min-board-coverage", type=float, default=0.015,
                        help="reject chessboards whose corner bounding box is smaller than this image fraction")
    parser.add_argument("--max-reprojection-px", type=float, default=2.0,
                        help="reject captures whose PnP RMS exceeds this value")
    parser.add_argument("--min-samples", type=int, default=15)
    parser.add_argument("--robot-id", default=None)
    parser.add_argument("--position-unit", choices=["m", "mm"], default="mm")
    parser.add_argument("--angle-unit", choices=["rad", "deg"], default="deg")
    parser.add_argument("--output-dir")
    parser.add_argument("--preview-host", default="0.0.0.0")
    parser.add_argument("--preview-port", type=int, default=8080)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--method", choices=["TSAI", "PARK", "HORAUD", "ANDREFF", "DANIILIDIS"], default="PARK")
    parser.add_argument("--no-solve", action="store_true")
    args = parser.parse_args()
    if args.camera_backend == "v4l2" and args.intrinsics is None:
        parser.error("--intrinsics is required when --camera-backend=v4l2")
    if args.min_samples < 8:
        parser.error("--min-samples must be at least 8")
    if not np.isfinite(args.tag_size_mm) or args.tag_size_mm <= 0:
        parser.error("--tag-size-mm must be positive")
    if args.chessboard_columns < 3 or args.chessboard_rows < 3:
        parser.error("--chessboard-columns/--chessboard-rows must be at least 3")
    if not np.isfinite(args.square_size_mm) or args.square_size_mm <= 0:
        parser.error("--square-size-mm must be positive")
    if not np.isfinite(args.max_reprojection_px) or args.max_reprojection_px <= 0:
        parser.error("--max-reprojection-px must be positive")
    if not 0.0 <= args.min_board_coverage <= 1.0:
        parser.error("--min-board-coverage must be between 0 and 1")
    if not 1 <= args.preview_port <= 65535:
        parser.error("--preview-port must be between 1 and 65535")
    try:
        data_dir = collect(args)
        print(f"\nCollection complete: {data_dir}")
        if not args.no_solve:
            solve_from_data(
                data_dir,
                args.tag_size_mm,
                args.method,
                target_type=args.target_type,
                chessboard_columns=args.chessboard_columns,
                chessboard_rows=args.chessboard_rows,
                square_size_mm=args.square_size_mm,
            )
    except (OSError, ValueError, RuntimeError, cv2.error) as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())