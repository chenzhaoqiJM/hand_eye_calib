#!/usr/bin/env python3
"""Calibrate a monocular USB V4L2 RGB camera with a browser preview."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
import threading

import cv2
import numpy as np

from calibration import calibrate_camera, chessboard_object_points, detect_chessboard
from camera import V4L2Camera


HERE = Path(__file__).resolve().parent


@dataclass
class FrameData:
    sequence: int
    raw: np.ndarray
    corners: np.ndarray | None
    jpeg: bytes
    valid: bool
    message: str
    sharpness: float
    coverage: float


class CalibrationApp:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.camera = V4L2Camera(
            args.device, args.width, args.height, args.fps, args.pixel_format
        )
        self.condition = threading.Condition()
        self.frame: FrameData | None = None
        self.error: str | None = None
        self.stopping = False
        self.samples: list[tuple[np.ndarray, np.ndarray]] = []
        self.result: dict[str, object] | None = None
        self.session_dir = Path(args.session_dir).resolve() if args.session_dir else (
            HERE / "captures" / datetime.now().strftime("%Y-%m-%d_%H%M%S")
        )
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)

    def start(self) -> None:
        self.camera.start()
        config = self.camera.config
        actual_size = (int(config["width"]), int(config["height"]))
        requested_size = (self.args.width, self.args.height)
        if actual_size != requested_size:
            self.camera.stop()
            raise RuntimeError(
                f"camera returned {actual_size[0]}x{actual_size[1]}, "
                f"requested {requested_size[0]}x{requested_size[1]}"
            )
        self.thread.start()
        with self.condition:
            ready = self.condition.wait_for(
                lambda: self.frame is not None or self.error is not None, timeout=10.0
            )
            if self.error:
                raise RuntimeError(self.error)
            if not ready:
                raise RuntimeError("timed out waiting for the first camera frame")

    def stop(self) -> None:
        with self.condition:
            self.stopping = True
            self.condition.notify_all()
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.camera.stop()

    def _capture_loop(self) -> None:
        sequence = 0
        try:
            while True:
                with self.condition:
                    if self.stopping:
                        return
                image = self.camera.read()
                detection = detect_chessboard(
                    image,
                    self.args.columns,
                    self.args.rows,
                    self.args.min_sharpness,
                    self.args.min_coverage,
                )
                ok, encoded = cv2.imencode(
                    ".jpg", detection.annotated, [cv2.IMWRITE_JPEG_QUALITY, 85]
                )
                if not ok:
                    raise RuntimeError("failed to encode preview frame")
                sequence += 1
                data = FrameData(
                    sequence,
                    image,
                    detection.corners,
                    encoded.tobytes(),
                    detection.valid,
                    detection.message,
                    detection.sharpness,
                    detection.coverage,
                )
                with self.condition:
                    self.frame = data
                    self.condition.notify_all()
        except BaseException as exc:
            with self.condition:
                self.error = str(exc)
                self.condition.notify_all()

    def wait_for_jpeg(self, previous: int) -> tuple[int, bytes] | None:
        with self.condition:
            self.condition.wait_for(
                lambda: (
                    (self.frame is not None and self.frame.sequence > previous)
                    or self.stopping
                    or self.error is not None
                )
            )
            if self.stopping or self.error or self.frame is None:
                return None
            return self.frame.sequence, self.frame.jpeg

    def status(self) -> dict[str, object]:
        with self.condition:
            frame = self.frame
            return {
                "valid": bool(frame and frame.valid),
                "message": self.error or (frame.message if frame else "waiting for camera"),
                "sharpness": frame.sharpness if frame else 0.0,
                "coverage": frame.coverage if frame else 0.0,
                "samples": len(self.samples),
                "min_samples": self.args.min_samples,
                "calibrated": self.result is not None,
                "output": str(Path(self.args.output).resolve()) if self.result else None,
            }

    def capture_sample(self) -> tuple[bool, str]:
        with self.condition:
            frame = self.frame
            if frame is None:
                return False, "no frame is available"
            if not frame.valid or frame.corners is None:
                return False, frame.message
            image = frame.raw.copy()
            corners = frame.corners.copy()
            index = len(self.samples)
            self.samples.append((chessboard_object_points(
                self.args.columns, self.args.rows, self.args.square_size_mm
            ), corners))
        path = self.session_dir / f"frame_{index:03d}.png"
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            with self.condition:
                self.samples.pop()
            return False, "failed to encode captured image"
        encoded.tofile(str(path))
        return True, f"captured sample {index + 1}"

    def run_calibration(self) -> tuple[bool, str]:
        with self.condition:
            samples = [(obj.copy(), img.copy()) for obj, img in self.samples]
        if len(samples) < self.args.min_samples:
            return False, f"need at least {self.args.min_samples} valid samples"
        object_points = [sample[0] for sample in samples]
        image_points = [sample[1] for sample in samples]
        rms, matrix, coefficients, per_view = calibrate_camera(
            object_points, image_points, (self.args.width, self.args.height)
        )
        result: dict[str, object] = {
            "fx": float(matrix[0, 0]),
            "fy": float(matrix[1, 1]),
            "cx": float(matrix[0, 2]),
            "cy": float(matrix[1, 2]),
            "width": int(self.args.width),
            "height": int(self.args.height),
            "coeffs": [float(value) for value in coefficients],
            "distortion_model": "plumb_bob",
        }
        output = Path(self.args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        report = {
            "rms_reprojection_error_px": rms,
            "per_view_error_px": per_view,
            "sample_count": len(samples),
            "board_inner_corners": [self.args.columns, self.args.rows],
            "square_size_mm": self.args.square_size_mm,
            "camera": self.camera.config,
            "intrinsics": result,
        }
        (self.session_dir / "calibration_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        with self.condition:
            self.result = result
        print("\nCalibration completed")
        print(f"RMS reprojection error: {rms:.6f} px")
        print("Camera matrix K:")
        print(np.array2string(matrix, precision=9, suppress_small=False))
        print("Distortion coefficients [k1, k2, p1, p2, k3, ...]:")
        print(np.array2string(coefficients, precision=9, suppress_small=False))
        print("Intrinsics JSON:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"Saved intrinsics: {output}")
        print(f"Saved report: {self.session_dir / 'calibration_report.json'}", flush=True)
        return True, f"calibration saved to {output}; RMS={rms:.4f} px"


PAGE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>单目 RGB 相机标定</title><style>
:root{color-scheme:dark}body{max-width:1100px;margin:0 auto;padding:22px;background:#111827;color:#e5e7eb;font:16px system-ui,sans-serif}
h1{font-size:25px}.panel{background:#1f2937;border:1px solid #374151;border-radius:12px;padding:16px;margin:14px 0}
img{display:block;width:100%;height:auto;border-radius:8px;background:#000}.row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
button{border:0;border-radius:8px;padding:11px 18px;font-weight:700;cursor:pointer;background:#2563eb;color:white}
button:disabled{opacity:.45;cursor:not-allowed}#calibrate{background:#059669}.valid{color:#4ade80}.invalid{color:#f87171}
code{color:#93c5fd}small{color:#9ca3af}</style></head><body>
<h1>单目 USB RGB 相机标定</h1><div class="panel"><img src="/stream.mjpg" alt="camera preview"></div>
<div class="panel"><div class="row"><strong id="state">正在连接…</strong><span id="quality"></span></div>
<p>有效样本：<b id="count">0</b> / <span id="minimum">0</span></p><div class="row">
<button id="capture" onclick="post('/api/capture')">采集当前有效图像</button>
<button id="calibrate" onclick="post('/api/calibrate')">计算并保存标定结果</button></div>
<p id="notice"></p><small>请从不同距离和角度拍摄，使棋盘覆盖画面中心、边缘和四角；避免连续采集相同姿态。</small></div>
<script>
async function refresh(){try{const s=await (await fetch('/api/status',{cache:'no-store'})).json();
const state=document.getElementById('state');state.textContent=s.message;state.className=s.valid?'valid':'invalid';
document.getElementById('quality').textContent=`清晰度 ${s.sharpness.toFixed(1)} · 覆盖率 ${(s.coverage*100).toFixed(1)}%`;
document.getElementById('count').textContent=s.samples;document.getElementById('minimum').textContent=s.min_samples;
document.getElementById('capture').disabled=!s.valid;document.getElementById('calibrate').disabled=s.samples<s.min_samples;
if(s.calibrated)document.getElementById('notice').textContent=`已保存：${s.output}`;}catch(e){document.getElementById('state').textContent=e;}}
async function post(url){const b=document.querySelectorAll('button');b.forEach(x=>x.disabled=true);try{const r=await fetch(url,{method:'POST'});const d=await r.json();document.getElementById('notice').textContent=d.message;}catch(e){document.getElementById('notice').textContent=e;}finally{refresh();}}
setInterval(refresh,500);refresh();</script></body></html>"""


def handler_for(app: CalibrationApp):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, status: HTTPStatus, value: dict[str, object]) -> None:
            body = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/":
                body = PAGE.encode()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/status":
                self._json(HTTPStatus.OK, app.status())
            elif self.path == "/stream.mjpg":
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                sequence = 0
                try:
                    while (item := app.wait_for_jpeg(sequence)) is not None:
                        sequence, jpeg = item
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                        self.wfile.write(jpeg + b"\r\n")
                except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                    pass
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            if self.path == "/api/capture":
                ok, message = app.capture_sample()
            elif self.path == "/api/calibrate":
                try:
                    ok, message = app.run_calibration()
                except (OSError, ValueError, cv2.error) as exc:
                    ok, message = False, f"calibration failed: {exc}"
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._json(HTTPStatus.OK if ok else HTTPStatus.BAD_REQUEST, {"ok": ok, "message": message})

        def log_message(self, *args: object) -> None:
            del args

    return Handler


def preview_urls(host: str, port: int) -> list[str]:
    if host not in ("", "0.0.0.0", "::"):
        return [f"http://{host}:{port}"]
    addresses = {"127.0.0.1"}
    try:
        addresses.update(socket.gethostbyname_ex(socket.gethostname())[2])
    except OSError:
        pass
    return [f"http://{address}:{port}" for address in sorted(addresses) if ":" not in address]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Browser-assisted V4L2 monocular RGB camera calibration")
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--pixel-format", default="MJPG", help="V4L2 FourCC, e.g. MJPG or YUYV")
    parser.add_argument("--columns", type=int, default=9, help="chessboard inner corner columns")
    parser.add_argument("--rows", type=int, default=6, help="chessboard inner corner rows")
    parser.add_argument("--square-size-mm", type=float, default=25.0)
    parser.add_argument("--min-samples", type=int, default=15)
    parser.add_argument("--min-sharpness", type=float, default=80.0)
    parser.add_argument("--min-coverage", type=float, default=0.05)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--session-dir")
    parser.add_argument("--output", default=str(HERE / "intrinsics.json"))
    args = parser.parse_args()
    if args.columns < 3 or args.rows < 3:
        parser.error("--columns and --rows must be at least 3")
    if args.square_size_mm <= 0:
        parser.error("--square-size-mm must be positive")
    if args.min_samples < 8:
        parser.error("--min-samples must be at least 8")
    if args.min_sharpness < 0 or not 0 <= args.min_coverage <= 1:
        parser.error("quality thresholds are invalid")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    return args


def main() -> int:
    args = parse_args()
    app = CalibrationApp(args)
    server: ThreadingHTTPServer | None = None
    try:
        app.start()
        server = ThreadingHTTPServer((args.host, args.port), handler_for(app))
        server.daemon_threads = True
        print(f"Camera: {app.camera.config}")
        print(f"Captures: {app.session_dir}")
        print("Open browser preview:")
        for url in preview_urls(args.host, int(server.server_address[1])):
            print(f"  {url}")
        print("Press Ctrl+C after calibration to exit.", flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping calibration server.")
    except (OSError, ValueError, RuntimeError, cv2.error) as exc:
        raise SystemExit(str(exc)) from exc
    finally:
        if server is not None:
            server.server_close()
        app.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
