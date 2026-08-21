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
    load_intrinsics_json,
    parse_pattern,
    reprojection_error,
    save_payload,
    solve_camera_plane_pose,
    solve_homography,
    undistort_image,
)


class LiveState:
    def __init__(
        self,
        pattern: tuple[int, int],
        square_size: float,
        output: Path,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        undistort: bool,
    ) -> None:
        self.pattern = pattern
        self.square_size = square_size
        self.output = output
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.undistort = undistort
        self.condition = threading.Condition()
        self.frame: np.ndarray | None = None
        self.jpeg: bytes | None = None
        self.sequence = 0
        self.corners: np.ndarray | None = None
        self.result: dict | None = None
        self.calibration_jpeg: bytes | None = None
        self.error: str | None = None
        self.stopping = False

    @property
    def calibrated(self) -> bool:
        with self.condition:
            return self.result is not None and self.calibration_jpeg is not None

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
        camera_plane, pnp_rms = solve_camera_plane_pose(
            corners, self.pattern, self.square_size,
            self.camera_matrix,
            np.zeros_like(self.dist_coeffs) if self.undistort else self.dist_coeffs,
        )
        errors = reprojection_error(matrix, corners, plane_points)
        payload = homography_payload(
            matrix, self.pattern, self.square_size,
            (frame.shape[1], frame.shape[0]), inliers, errors,
        )
        payload["plane_origin_pixel"] = {
            "u": float(corners[0, 0]),
            "v": float(corners[0, 1]),
        }
        payload["matrix_camera_plane"] = camera_plane.tolist()
        payload["camera_coordinate_unit"] = payload["plane_coordinate_unit"]
        payload["pnp_reprojection_rms"] = pnp_rms
        payload["undistorted"] = self.undistort
        self.output.parent.mkdir(parents=True, exist_ok=True)
        save_payload(self.output, payload)
        annotated = draw_detection(frame, corners, self.pattern)
        origin = tuple(np.round(corners[0]).astype(int))
        cv2.drawMarker(
            annotated, origin, (0, 0, 255), cv2.MARKER_CROSS, 36, 3,
        )
        cv2.putText(
            annotated, "origin (0, 0)", (origin[0] + 12, origin[1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA,
        )
        ok, encoded = cv2.imencode(
            ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90]
        )
        if not ok:
            raise RuntimeError("could not encode calibration image")
        with self.condition:
            self.result = payload
            self.calibration_jpeg = encoded.tobytes()
            self.error = None
        return payload


def make_handler(state: LiveState):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = self.path.partition("?")[0]
            if path == "/":
                body = self._page().encode()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/stream.mjpg":
                self._stream()
                return
            if path == "/calibration.jpg":
                with state.condition:
                    image = state.calibration_jpeg
                if image is None:
                    self.send_error(HTTPStatus.NOT_FOUND, "calibration not finished")
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(image)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(image)
                return
            if path == "/status":
                with state.condition:
                    response = {
                        "detected": state.corners is not None,
                        "calibrated": state.result is not None,
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
            return r"""<!doctype html>
<meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Planar Homography</title>
<style>
*{box-sizing:border-box}[hidden]{display:none!important}body{margin:0;background:#17191c;color:#eee;font:16px system-ui,sans-serif}
main{max-width:1200px;margin:auto;padding:18px}h1{font-size:22px;font-weight:500;text-align:center}
#viewport{position:relative;margin:12px auto;background:#000;line-height:0;width:fit-content;max-width:100%}
#stream,#calibration{display:block;max-width:100%;height:auto}#calibration{cursor:crosshair}
#overlay{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
.toolbar{display:flex;justify-content:center;align-items:center;gap:10px;flex-wrap:wrap}
button{padding:10px 16px;font-size:15px;cursor:pointer}.active{outline:2px solid #53a8ff}
#status{text-align:center;min-height:24px;margin:12px}.measurements{display:grid;grid-template-columns:1fr 1fr;gap:12px}
pre{margin:0;text-align:left;overflow:auto;background:#25282d;padding:12px;min-height:96px}
@media(max-width:700px){.measurements{grid-template-columns:1fr}}
</style><main><h1>Chessboard pixel-to-plane mapping</h1>
<div id="viewport"><img id="stream" src="/stream.mjpg" alt="camera stream">
<img id="calibration" hidden alt="calibration image"><canvas id="overlay" hidden></canvas></div>
<div class="toolbar"><button id="calculate">Calculate mapping matrix</button>
<button id="pointMode" hidden>Measure point</button><button id="distanceMode" hidden>Measure distance</button>
<button id="clear" hidden>Clear marks</button></div><div id="status">Waiting for chessboard...</div>
<div class="measurements"><pre id="measurement">Click Calculate to calibrate.</pre><pre id="result"></pre></div></main>
<script>
const $=s=>document.querySelector(s),status=$('#status'),result=$('#result'),measurement=$('#measurement');
const stream=$('#stream'),image=$('#calibration'),canvas=$('#overlay'),ctx=canvas.getContext('2d');
const pointButton=$('#pointMode'),distanceButton=$('#distanceMode'),clearButton=$('#clear');
let H=null,mode='point',points=[],calibrated=false,coordinateUnit='unknown';
function mapPixel(u,v){const x=H[0][0]*u+H[0][1]*v+H[0][2],y=H[1][0]*u+H[1][1]*v+H[1][2],w=H[2][0]*u+H[2][1]*v+H[2][2];return{x:x/w,y:y/w}}
function redraw(){ctx.clearRect(0,0,canvas.width,canvas.height);ctx.lineWidth=Math.max(2,canvas.width/500);ctx.font=`${Math.max(14,canvas.width/55)}px system-ui`;
 if(points.length===2){ctx.strokeStyle='#ffd54f';ctx.beginPath();ctx.moveTo(points[0].u,points[0].v);ctx.lineTo(points[1].u,points[1].v);ctx.stroke()}
 points.forEach((p,i)=>{ctx.fillStyle=i?'#53a8ff':'#ffd54f';ctx.beginPath();ctx.arc(p.u,p.v,Math.max(5,canvas.width/150),0,Math.PI*2);ctx.fill();ctx.fillText(i?'B':'A',p.u+9,p.v-9)})}
function report(){if(!points.length){measurement.textContent=mode==='point'?'Click the image to measure a point.':'Click two points to measure distance.';return}
 let text=points.map((p,i)=>`${i?'B':'A'} pixel: (${p.u.toFixed(2)}, ${p.v.toFixed(2)})\n${i?'B':'A'} plane [${coordinateUnit}]: (${p.x.toFixed(3)}, ${p.y.toFixed(3)})\n${i?'B':'A'} camera [${coordinateUnit}]: (${p.camera[0].toFixed(3)}, ${p.camera[1].toFixed(3)}, ${p.camera[2].toFixed(3)})`).join('\n\n');
 if(points.length===2){const planeDistance=Math.hypot(points[1].x-points[0].x,points[1].y-points[0].y);const cameraDistance=Math.hypot(...points[1].camera.map((value,index)=>value-points[0].camera[index]));text+=`\n\nPlane distance A-B [${coordinateUnit}]: ${planeDistance.toFixed(3)}\nCamera distance A-B [${coordinateUnit}]: ${cameraDistance.toFixed(3)}`};measurement.textContent=text}
function setMode(next){mode=next;points=[];pointButton.classList.toggle('active',mode==='point');distanceButton.classList.toggle('active',mode==='distance');redraw();report()}
function mapCamera(x,y){const T=runtimeCameraTransform;return[T[0][0]*x+T[0][1]*y+T[0][3],T[1][0]*x+T[1][1]*y+T[1][3],T[2][0]*x+T[2][1]*y+T[2][3]]}
let runtimeCameraTransform=null;
function showMeasurement(r){H=r.matrix_pixel_to_plane;runtimeCameraTransform=r.matrix_camera_plane;coordinateUnit=r.plane_coordinate_unit||'unknown';calibrated=true;stream.hidden=true;image.hidden=false;canvas.hidden=false;
 image.onload=()=>{canvas.width=image.naturalWidth;canvas.height=image.naturalHeight;redraw()};image.src='/calibration.jpg?t='+Date.now();
 [pointButton,distanceButton,clearButton].forEach(b=>b.hidden=false);setMode('point');result.textContent=JSON.stringify(r,null,2)}
image.onclick=e=>{if(!H||!runtimeCameraTransform)return;const rect=image.getBoundingClientRect(),u=(e.clientX-rect.left)*image.naturalWidth/rect.width,v=(e.clientY-rect.top)*image.naturalHeight/rect.height,p=mapPixel(u,v),camera=mapCamera(p.x,p.y);
 if(mode==='point')points=[{u,v,...p,camera}];else{if(points.length>=2)points=[];points.push({u,v,...p,camera})}redraw();report()};
pointButton.onclick=()=>setMode('point');distanceButton.onclick=()=>setMode('distance');clearButton.onclick=()=>{points=[];redraw();report()};
async function poll(){try{const r=await fetch('/status',{cache:'no-store'}),s=await r.json();
 if(!calibrated)status.textContent=s.error|| (s.detected?'Chessboard detected. Ready to calculate.':'Chessboard not detected.');
 }catch(e){status.textContent='Connection error: '+e} setTimeout(poll,500)}
 $('#calculate').onclick=async()=>{status.textContent='Calculating...';
 const r=await fetch('/calculate',{method:'POST'}),s=await r.json();
 status.textContent=s.ok?'Calibration saved. Click the image to measure.':s.error;if(s.result)showMeasurement(s.result)};poll();
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
    parser.add_argument(
        "--intrinsics", type=Path,
        default=Path(__file__).resolve().parent.parent
        / "monocular_rgb_calibration" / "intrinsics.json",
        help="camera intrinsics JSON (default: ../monocular_rgb_calibration/intrinsics.json)",
    )
    parser.add_argument(
        "--undistort", action=argparse.BooleanOptionalAction, default=True,
        help="undistort frames before detection (default: true)",
    )
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
    actual_size = (
        int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
        int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
    )
    if actual_size != (args.width, args.height):
        capture.release()
        raise SystemExit(
            f"camera returned resolution {actual_size[0]}x{actual_size[1]}, "
            f"but --width/--height requested {args.width}x{args.height}; "
            "use matching camera settings or update the intrinsics file"
        )
    try:
        camera_matrix, dist_coeffs = load_intrinsics_json(args.intrinsics, actual_size)
    except ValueError as exc:
        capture.release()
        raise SystemExit(f"Invalid intrinsics: {exc}") from exc
    state = LiveState(
        pattern, args.square_size, args.output, camera_matrix, dist_coeffs,
        args.undistort,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    server.timeout = 0.05
    print(f"Open http://127.0.0.1:{server.server_address[1]} in a browser")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("failed to read camera frame")
            working_frame = (
                undistort_image(frame, camera_matrix, dist_coeffs)
                if args.undistort else frame
            )
            state.update(working_frame, detect_chessboard(working_frame, pattern))
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
