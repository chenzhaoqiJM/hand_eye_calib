#!/usr/bin/env python3
"""Live chessboard detection with a browser button to solve homography."""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading

import cv2
import numpy as np

from homography_common import (
    detect_chessboard,
    draw_detection,
    homography_payload,
    parse_pattern,
    reprojection_error,
    save_payload,
    solve_homography,
)


class LiveState:
    def __init__(self, pattern: tuple[int, int], square_size: float, output: Path) -> None:
        self.pattern = pattern
        self.square_size = square_size
        self.output = output
        self.condition = threading.Condition()
        self.frame: np.ndarray | None = None
        self.jpeg: bytes | None = None
        self.sequence = 0
        self.corners: np.ndarray | None = None
        self.result: dict | None = None
        self.error: str | None = None
        self.stopping = False

    def update(self, frame: np.ndarray, corners: np.ndarray | None) -> None:
        annotated = draw_detection(frame, corners, self.pattern)
        ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise RuntimeError("could not encode preview frame")
        with self.condition:
            self.frame = frame
            self.jpeg = encoded.tobytes()
            self.corners = corners
            self.sequence += 1
            self.condition.notify_all()

    def calculate(self) -> dict:
        with self.condition:
            corners = None if self.corners is None else self.corners.copy()
            frame = None if self.frame is None else self.frame.copy()
        if corners is None or frame is None:
            raise ValueError("no valid chessboard is currently detected")
        matrix, plane_points, inliers = solve_homography(
            corners, self.pattern, self.square_size
        )
        errors = reprojection_error(matrix, corners, plane_points)
        payload = homography_payload(
            matrix, self.pattern, self.square_size,
            (frame.shape[1], frame.shape[0]), inliers, errors,
        )
        self.output.parent.mkdir(parents=True, exist_ok=True)
        save_payload(self.output, payload)
        with self.condition:
            self.result = payload
            self.error = None
        return payload


def make_handler(state: LiveState):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/":
                body = self._page().encode()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/stream.mjpg":
                self._stream()
                return
            if self.path == "/status":
                with state.condition:
                    response = {
                        "detected": state.corners is not None,
                        "result": state.result,
                        "error": state.error,
                    }
                self._json(response)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            if self.path != "/calculate":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                payload = state.calculate()
                self._json({"ok": True, "result": payload})
            except (RuntimeError, ValueError, OSError, cv2.error) as exc:
                with state.condition:
                    state.error = str(exc)
                self._json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)

        def _stream(self) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            sequence = 0
            try:
                while True:
                    with state.condition:
                        ready = state.condition.wait_for(
                            lambda: state.sequence > sequence or state.stopping,
                            timeout=10.0,
                        )
                        if state.stopping:
                            return
                        if not ready or state.jpeg is None:
                            continue
                        sequence = state.sequence
                        jpeg = state.jpeg
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                    self.wfile.write(jpeg + b"\r\n")
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                return

        def _json(self, value: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        @staticmethod
        def _page() -> str:
            return """<!doctype html>
<meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Planar Homography</title>
<style>
body{margin:0;background:#17191c;color:#eee;font:16px system-ui,sans-serif;text-align:center}
main{max-width:1100px;margin:auto;padding:18px}h1{font-size:22px;font-weight:500}
img{display:block;max-width:100%;height:auto;margin:12px auto;background:#000}
button{padding:10px 18px;font-size:16px;cursor:pointer}#status{min-height:24px;margin:12px}
pre{text-align:left;overflow:auto;background:#25282d;padding:12px}
</style><main><h1>Chessboard pixel-to-plane mapping</h1>
<img src="/stream.mjpg" alt="camera stream"><button id="calculate">Calculate mapping matrix</button>
<div id="status">Waiting for chessboard...</div><pre id="result"></pre></main>
<script>
const status=document.querySelector('#status'), result=document.querySelector('#result');
async function poll(){try{const r=await fetch('/status',{cache:'no-store'}),s=await r.json();
 status.textContent=s.error|| (s.detected?'Chessboard detected. Ready to calculate.':'Chessboard not detected.');
 if(s.result) result.textContent=JSON.stringify(s.result,null,2);
 }catch(e){status.textContent='Connection error: '+e} setTimeout(poll,500)}
 document.querySelector('#calculate').onclick=async()=>{status.textContent='Calculating...';
 const r=await fetch('/calculate',{method:'POST'}),s=await r.json();
 status.textContent=s.ok?'Saved mapping matrix.':s.error; if(s.result) result.textContent=JSON.stringify(s.result,null,2)}; poll();
</script>"""

        def log_message(self, *args) -> None:
            del args

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description="Live chessboard homography web tool")
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--pattern", default="9x6")
    parser.add_argument("--square-size", type=float, required=True)
    parser.add_argument("--output", type=Path, default=Path("pixel_to_plane_homography.json"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    pattern = parse_pattern(args.pattern)
    if args.square_size <= 0 or not np.isfinite(args.square_size):
        parser.error("--square-size must be positive")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    capture = cv2.VideoCapture(args.device, cv2.CAP_V4L2 if args.device.startswith("/dev/") else 0)
    if not capture.isOpened():
        raise SystemExit(f"could not open camera: {args.device}")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    capture.set(cv2.CAP_PROP_FPS, args.fps)
    state = LiveState(pattern, args.square_size, args.output)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    server.timeout = 0.05
    print(f"Open http://127.0.0.1:{server.server_address[1]} in a browser")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("failed to read camera frame")
            state.update(frame, detect_chessboard(frame, pattern))
            server.handle_request()
    except KeyboardInterrupt:
        pass
    finally:
        with state.condition:
            state.stopping = True
            state.condition.notify_all()
        server.server_close()
        capture.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
