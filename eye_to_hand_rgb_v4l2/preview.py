"""Threaded camera capture and browser-based MJPEG preview."""

from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import socket
import threading

import cv2
import numpy as np


class FrameStream:
    def __init__(self, camera) -> None:
        self._camera = camera
        self._condition = threading.Condition()
        self._frame: np.ndarray | None = None
        self._jpeg: bytes | None = None
        self._sequence = 0
        self._stopping = False
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()
        with self._condition:
            ready = self._condition.wait_for(
                lambda: self._frame is not None or self._error is not None, timeout=10.0
            )
            if self._error is not None:
                raise RuntimeError("Camera stream failed to start") from self._error
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
                raise RuntimeError("Camera stream stopped unexpectedly") from self._error
            if self._frame is None:
                raise RuntimeError("No camera frame is available")
            return self._frame.copy()

    def wait_for_jpeg(self, last_sequence: int) -> tuple[int, bytes] | None:
        with self._condition:
            self._condition.wait_for(
                lambda: self._sequence > last_sequence or self._stopping or self._error
            )
            if self._stopping or self._error or self._jpeg is None:
                return None
            return self._sequence, self._jpeg

    def _run(self) -> None:
        try:
            while True:
                with self._condition:
                    if self._stopping:
                        return
                frame = self._camera.capture()
                ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ok:
                    raise RuntimeError("Could not encode preview frame")
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
    def __init__(self, stream: FrameStream, host: str, port: int) -> None:
        self._server = ThreadingHTTPServer((host, port), self._handler(stream))
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

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
    def _handler(stream: FrameStream):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path == "/":
                    body = (
                        "<!doctype html><meta charset='utf-8'><meta name='viewport' "
                        "content='width=device-width'><title>Eye-to-hand preview</title>"
                        "<style>body{margin:0;background:#151719;color:#fff;font:16px "
                        "sans-serif;text-align:center}h1{font-size:20px;font-weight:500}"
                        "img{display:block;max-width:100%;height:auto;margin:auto}</style>"
                        "<h1>Eye-to-hand RGB preview</h1><img src='/stream.mjpg'>"
                    ).encode()
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path != "/stream.mjpg":
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                sequence = 0
                try:
                    while (item := stream.wait_for_jpeg(sequence)) is not None:
                        sequence, jpeg = item
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                        self.wfile.write(jpeg + b"\r\n")
                except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                    pass

            def log_message(self, *args) -> None:
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