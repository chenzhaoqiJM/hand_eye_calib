#!/usr/bin/env python3
"""Collect camera images and manually entered flange poses, then solve hand-eye.

The robot is never connected by this script. Move the robot manually, read the
flange pose from the teach pendant/controller, enter it here, and repeat.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
import threading

import cv2
import numpy as np

from calibrate_from_data import normalize_pose_units, solve_from_data


HERE = Path(__file__).resolve().parent
DEFAULT_TAG_SIZE_MM = 50.0
DEFAULT_MIN_SAMPLES = 12


class RetakeSample(Exception):
    """Raised when the operator wants to discard the just-captured frame."""


class FrameStream:
    """Read the camera in one thread and publish its latest frame."""

    def __init__(self, camera) -> None:
        self._camera = camera
        self._condition = threading.Condition()
        self._frame: np.ndarray | None = None
        self._jpeg: bytes | None = None
        self._sequence = 0
        self._stopping = False
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run, name="camera-frame-stream", daemon=True
        )

    def start(self) -> None:
        self._thread.start()
        with self._condition:
            ready = self._condition.wait_for(
                lambda: self._frame is not None or self._error is not None,
                timeout=10.0,
            )
            if self._error is not None:
                raise RuntimeError("Camera preview failed to start") from self._error
            if not ready:
                raise RuntimeError("Timed out waiting for the first camera frame")

    def stop(self) -> None:
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        self._thread.join(timeout=2.0)

    def snapshot(self) -> np.ndarray:
        with self._condition:
            if self._error is not None:
                raise RuntimeError("Camera preview stopped unexpectedly") from self._error
            if self._frame is None:
                raise RuntimeError("No camera frame is available")
            return self._frame.copy()

    def wait_for_jpeg(self, last_sequence: int) -> tuple[int, bytes] | None:
        with self._condition:
            self._condition.wait_for(
                lambda: (
                    self._sequence > last_sequence
                    or self._stopping
                    or self._error is not None
                )
            )
            if self._stopping or self._error is not None or self._jpeg is None:
                return None
            if self._sequence <= last_sequence:
                return None
            return self._sequence, self._jpeg

    def _run(self) -> None:
        try:
            while True:
                with self._condition:
                    if self._stopping:
                        return
                frame = self._camera.capture()
                ok, encoded = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85]
                )
                if not ok:
                    raise RuntimeError("Could not encode camera preview frame")
                with self._condition:
                    self._frame = frame
                    self._jpeg = encoded.tobytes()
                    self._sequence += 1
                    self._condition.notify_all()
        except BaseException as exc:
            with self._condition:
                self._error = exc
                self._condition.notify_all()


class PreviewServer:
    """Serve an MJPEG camera preview using only the Python standard library."""

    def __init__(self, frame_stream: FrameStream, host: str, port: int) -> None:
        handler = self._make_handler(frame_stream)
        self._server = ThreadingHTTPServer((host, port), handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="camera-preview-http",
            daemon=True,
        )

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2.0)

    @staticmethod
    def _make_handler(frame_stream: FrameStream):
        class PreviewHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path == "/":
                    body = (
                        "<!doctype html><html><head><meta charset='utf-8'>"
                        "<meta name='viewport' content='width=device-width'>"
                        "<title>Hand-eye camera preview</title>"
                        "<style>body{margin:0;background:#151719;color:#f4f4f4;"
                        "font:16px sans-serif;text-align:center}h1{font-size:20px;"
                        "font-weight:500}img{display:block;max-width:100%;height:auto;"
                        "margin:auto}</style></head><body>"
                        "<h1>Hand-eye camera preview</h1>"
                        "<img src='/stream.mjpg' alt='Live camera preview'>"
                        "</body></html>"
                    ).encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path == "/stream.mjpg":
                    self.send_response(HTTPStatus.OK)
                    self.send_header(
                        "Content-Type",
                        "multipart/x-mixed-replace; boundary=frame",
                    )
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    sequence = 0
                    try:
                        while True:
                            item = frame_stream.wait_for_jpeg(sequence)
                            if item is None:
                                return
                            sequence, jpeg = item
                            self.wfile.write(b"--frame\r\n")
                            self.wfile.write(b"Content-Type: image/jpeg\r\n")
                            self.wfile.write(
                                f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                            )
                            self.wfile.write(jpeg)
                            self.wfile.write(b"\r\n")
                    except (
                        BrokenPipeError,
                        ConnectionAbortedError,
                        ConnectionResetError,
                    ):
                        pass
                    return
                if self.path == "/favicon.ico":
                    self.send_response(HTTPStatus.NO_CONTENT)
                    self.end_headers()
                    return
                self.send_error(HTTPStatus.NOT_FOUND)

            def log_message(self, *args) -> None:
                del args
                return

        return PreviewHandler


def local_preview_urls(host: str, port: int) -> list[str]:
    if host not in ("0.0.0.0", "::", ""):
        return [f"http://{host}:{port}"]
    addresses = {"127.0.0.1"}
    try:
        addresses.update(socket.gethostbyname_ex(socket.gethostname())[2])
    except OSError:
        pass
    return [
        f"http://{address}:{port}"
        for address in sorted(addresses, key=lambda value: value.startswith("127."))
        if ":" not in address
    ]


def save_intrinsics(camera, save_dir: Path) -> None:
    k = camera.intrinsics
    payload = {
        "fx": float(k[0, 0]),
        "fy": float(k[1, 1]),
        "cx": float(k[0, 2]),
        "cy": float(k[1, 2]),
        "width": int(camera.resolution[0]),
        "height": int(camera.resolution[1]),
        "coeffs": camera.dist_coeffs.tolist(),
        "distortion_model": camera.distortion_model,
    }
    (save_dir / "intrinsics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def save_session(camera, save_dir: Path, args: argparse.Namespace) -> None:
    payload = {
        "robot_id": args.robot_id,
        "camera_mount": "eye_in_hand",
        "output_transform": "T_flange_camera",
        "tag_family": "DICT_APRILTAG_36h11",
        "tag_size_mm": float(args.tag_size_mm),
        "pose_source": "manual",
        "pose_position_unit": "m",
        "pose_angle_unit": "rad",
        "camera_device": camera.device_info,
        "camera_stream": camera.stream_config,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    (save_dir / "session.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def write_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"Could not encode image: {path}")
    encoded.tofile(str(path))


def parse_pose_line(line: str, position_unit: str, angle_unit: str) -> np.ndarray:
    parts = line.replace(",", " ").split()
    if len(parts) != 6:
        raise ValueError("Enter exactly 6 numbers: x y z rx ry rz")
    values = np.asarray(parts, dtype=np.float64)
    return normalize_pose_units(values, position_unit, angle_unit)


def save_pose(save_dir: Path, index: int, pose_m_rad: np.ndarray) -> None:
    payload = {
        "x": float(pose_m_rad[0]),
        "y": float(pose_m_rad[1]),
        "z": float(pose_m_rad[2]),
        "rx": float(pose_m_rad[3]),
        "ry": float(pose_m_rad[4]),
        "rz": float(pose_m_rad[5]),
        "position_unit": "m",
        "angle_unit": "rad",
    }
    path = save_dir / f"pose_{index:03d}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_pose_text(save_dir: Path, index: int, pose_m_rad: np.ndarray) -> None:
    path = save_dir / "poses.txt"
    if not path.exists():
        path.write_text(
            "# idx image x_m y_m z_m rx_rad ry_rad rz_rad rx_deg ry_deg rz_deg\n",
            encoding="utf-8",
        )
    rx_deg, ry_deg, rz_deg = np.degrees(pose_m_rad[3:])
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            f"{index} frame_{index:03d}.png "
            f"{pose_m_rad[0]:.9f} {pose_m_rad[1]:.9f} {pose_m_rad[2]:.9f} "
            f"{pose_m_rad[3]:.9f} {pose_m_rad[4]:.9f} {pose_m_rad[5]:.9f} "
            f"{rx_deg:.6f} {ry_deg:.6f} {rz_deg:.6f}\n"
        )


def prompt_pose(index: int, position_unit: str, angle_unit: str) -> np.ndarray | None:
    while True:
        line = input(
            f"[{index:03d}] Enter flange pose x y z rx ry rz "
            f"({position_unit}, {angle_unit}); q=finish, r=retake: "
        ).strip()
        if line.lower() == "q":
            return None
        if line.lower() == "r":
            raise RetakeSample
        try:
            return parse_pose_line(line, position_unit, angle_unit)
        except ValueError as exc:
            print(f"  Invalid pose: {exc}")


def collect(args: argparse.Namespace) -> Path:
    from camera import Camera

    save_dir = Path(args.output_dir).resolve() if args.output_dir else (
        HERE / "data" / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    camera = Camera(width=args.width, height=args.height, fps=args.fps)
    camera_started = False
    frame_stream: FrameStream | None = None
    preview_server: PreviewServer | None = None
    try:
        camera.start()
        camera_started = True
        save_intrinsics(camera, save_dir)
        save_session(camera, save_dir, args)
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
            print("Keep this terminal open and allow Python through Windows Firewall.")
        print("Move robot to varied poses. Keep exactly one AprilTag visible.")
        print("Press Enter to capture image, then type the matching flange pose.")

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

            image = frame_stream.snapshot() if frame_stream else camera.capture()
            preview_path = save_dir / f"frame_{index:03d}.png"
            write_image(preview_path, image)
            print(f"  Saved image: {preview_path.name}")

            try:
                pose = prompt_pose(index, args.position_unit, args.angle_unit)
            except RetakeSample:
                preview_path.unlink(missing_ok=True)
                print("  Retake requested; image removed.")
                continue
            if pose is None:
                preview_path.unlink(missing_ok=True)
                if index < args.min_samples:
                    print(f"Need at least {args.min_samples} samples, currently {index}.")
                    continue
                break

            save_pose(save_dir, index, pose)
            append_pose_text(save_dir, index, pose)
            print(
                "  Saved pose: "
                f"XYZ=({pose[0]:.4f}, {pose[1]:.4f}, {pose[2]:.4f}) m, "
                f"RPY=({np.degrees(pose[3]):.2f}, "
                f"{np.degrees(pose[4]):.2f}, {np.degrees(pose[5]):.2f}) deg"
            )
            index += 1
    finally:
        if preview_server is not None:
            preview_server.stop()
        if frame_stream is not None:
            frame_stream.stop()
        if camera_started:
            camera.stop()

    return save_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Manual hand-eye collection: D405 images + typed flange poses"
    )
    parser.add_argument("--tag-size-mm", type=float, default=DEFAULT_TAG_SIZE_MM)
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES)
    parser.add_argument("--robot-id", type=int, choices=[0, 1], default=1)
    parser.add_argument("--position-unit", choices=["m", "mm"], default="mm")
    parser.add_argument("--angle-unit", choices=["rad", "deg"], default="deg")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--preview-host",
        default="0.0.0.0",
        help="Browser preview bind address (default: all network interfaces)",
    )
    parser.add_argument(
        "--preview-port",
        type=int,
        default=8080,
        help="Browser preview TCP port (default: 8080)",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable the browser camera preview",
    )
    parser.add_argument(
        "--no-solve",
        action="store_true",
        help="Only collect data; solve later with calibrate_from_data.py",
    )
    args = parser.parse_args()

    if args.min_samples < 8:
        parser.error("--min-samples must be at least 8")
    if not np.isfinite(args.tag_size_mm) or args.tag_size_mm <= 0:
        parser.error("--tag-size-mm must be positive")
    if not 1 <= args.preview_port <= 65535:
        parser.error("--preview-port must be between 1 and 65535")

    data_dir = collect(args)
    print(f"\nCollection complete: {data_dir}")
    if not args.no_solve:
        solve_from_data(data_dir, args.tag_size_mm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
