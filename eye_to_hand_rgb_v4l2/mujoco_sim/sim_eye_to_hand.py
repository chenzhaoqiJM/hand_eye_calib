#!/usr/bin/env python3
"""Interactive MuJoCo eye-to-hand RGB calibration simulator."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import queue
import random
import sys
import threading
import time
from urllib.error import URLError
from urllib.request import urlopen
import xml.etree.ElementTree as ET

import cv2
from flask import Flask, Response, jsonify, redirect, render_template_string, request, url_for
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from calibrate_from_data import (  # noqa: E402
    METHODS,
    evaluate_matrix,
    load_robot_pose,
    solve_matrix,
)
from camera_model import reprojection_rms_px, solve_pnp  # noqa: E402


MENAGERIE_REPO = "https://github.com/google-deepmind/mujoco_menagerie"
MENAGERIE_RAW = "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main"
PANDA_DIR = "franka_emika_panda"
TARGET_TYPE = "chessboard"
DEFAULT_JOINTS = np.array([0.0, -0.55, 0.0, -2.15, 0.0, 1.75, 0.78], dtype=np.float64)
JOINT_LIMITS = np.array([
    [-2.70, 2.70],
    [-1.55, 1.20],
    [-2.70, 2.70],
    [-2.80, -0.20],
    [-2.70, 2.70],
    [0.15, 3.55],
    [-2.70, 2.70],
], dtype=np.float64)


INDEX_HTML = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MuJoCo Eye-to-hand RGB 采集</title>
  <style>
    :root{color-scheme:dark light}
    body{margin:0;font:14px/1.45 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#101214;color:#eef1f3}
    main{display:grid;grid-template-columns:minmax(320px,1fr) 360px;gap:16px;min-height:100vh;padding:16px;box-sizing:border-box}
    .video{min-width:0;background:#050607;border:1px solid #2a3036;border-radius:8px;overflow:hidden;display:flex;align-items:center;justify-content:center}
    img{display:block;width:100%;height:auto}
    aside{display:flex;flex-direction:column;gap:12px}
    section{border:1px solid #2a3036;border-radius:8px;padding:12px;background:#171a1d}
    h1,h2{margin:0 0 10px;font-size:16px;font-weight:650}
    h2{font-size:14px;color:#c9d1d9}
    .status{display:grid;grid-template-columns:1fr 1fr;gap:8px}
    .metric{padding:8px;border-radius:6px;background:#22272d}
    .metric b{display:block;font-size:12px;color:#98a3ad;font-weight:500}
    .metric span{font-size:18px}
    .ok{color:#65d184}.bad{color:#ff786e}.warn{color:#ffd36a}
    .row{display:grid;grid-template-columns:46px 1fr 68px;gap:8px;align-items:center;margin:8px 0}
    input[type=range]{width:100%}
    input[type=number]{width:64px;background:#0f1215;color:#eef1f3;border:1px solid #3a424a;border-radius:6px;padding:5px}
    button{appearance:none;border:1px solid #3a424a;background:#242a30;color:#eef1f3;border-radius:6px;padding:8px 10px;font-weight:650;cursor:pointer}
    button.primary{background:#2563eb;border-color:#3b82f6}
    button.danger{background:#66312f;border-color:#8d4541}
    .buttons{display:grid;grid-template-columns:1fr 1fr;gap:8px}
    code{color:#a9d1ff}
    @media (max-width: 860px){main{grid-template-columns:1fr}.video{align-items:start}}
  </style>
</head>
<body>
<main>
  <div class="video"><img src="/stream.mjpg" alt="preview"></div>
  <aside>
    <section>
      <h1>MuJoCo Eye-to-hand RGB 采集</h1>
      <div class="status">
        <div class="metric"><b>样本数</b><span id="samples">0</span></div>
        <div class="metric"><b>有效性</b><span id="valid" class="warn">等待</span></div>
        <div class="metric"><b>重投影</b><span id="rms">-</span></div>
        <div class="metric"><b>覆盖率</b><span id="area">-</span></div>
      </div>
    </section>
    <section>
      <h2>关节目标</h2>
      <div id="joints"></div>
      <div class="buttons">
        <button onclick="randomPose()">随机姿态</button>
        <button onclick="homePose()">回到初始</button>
      </div>
    </section>
    <section>
      <h2>采集与求解</h2>
      <div class="buttons">
        <button class="primary" onclick="capture()">采集当前帧</button>
        <button onclick="solve()">求解标定</button>
      </div>
      <p id="message"></p>
      <p>数据目录：<code id="dataDir"></code></p>
    </section>
  </aside>
</main>
<script>
const limits = {{ limits|tojson }};
const jointBox = document.getElementById("joints");
for (let i = 0; i < limits.length; i++) {
  const row = document.createElement("div");
  row.className = "row";
  row.innerHTML = `<label>q${i+1}</label><input id="q${i}" type="range" min="${limits[i][0]}" max="${limits[i][1]}" step="0.01"><input id="n${i}" type="number" step="0.01">`;
  jointBox.appendChild(row);
}
function setMessage(text){document.getElementById("message").textContent=text || "";}
function applyJoints(values){
  for (let i = 0; i < values.length; i++) {
    document.getElementById(`q${i}`).value = values[i].toFixed(3);
    document.getElementById(`n${i}`).value = values[i].toFixed(3);
  }
}
async function postJSON(path, body){
  const res = await fetch(path,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body || {})});
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}
for (let i = 0; i < limits.length; i++) {
  const range = document.getElementById(`q${i}`);
  const num = document.getElementById(`n${i}`);
  const send = async () => {
    const values = [...Array(limits.length).keys()].map(j => Number(document.getElementById(`q${j}`).value));
    num.value = range.value;
    await postJSON("/api/joints", {joints: values});
  };
  range.addEventListener("input", send);
  num.addEventListener("change", async () => { range.value = num.value; await send(); });
}
async function refresh(){
  const s = await (await fetch("/api/status")).json();
  applyJoints(s.joints);
  document.getElementById("samples").textContent = s.sample_count;
  document.getElementById("dataDir").textContent = s.data_dir;
  const valid = document.getElementById("valid");
  valid.textContent = s.quality.valid ? "有效" : (s.quality.reason || "无效");
  valid.className = s.quality.valid ? "ok" : "bad";
  document.getElementById("rms").textContent = s.quality.reprojection_rms_px == null ? "-" : `${s.quality.reprojection_rms_px.toFixed(3)} px`;
  document.getElementById("area").textContent = s.quality.coverage == null ? "-" : `${(s.quality.coverage * 100).toFixed(1)}%`;
}
async function randomPose(){ const r = await postJSON("/api/random_pose", {}); applyJoints(r.joints); }
async function homePose(){ const r = await postJSON("/api/home", {}); applyJoints(r.joints); }
async function capture(){
  try { const r = await postJSON("/api/capture", {}); setMessage(r.message); await refresh(); }
  catch(e){ setMessage(e.message); }
}
async function solve(){
  try { const r = await postJSON("/api/solve", {}); setMessage(r.message); await refresh(); }
  catch(e){ setMessage(e.message); }
}
refresh();
setInterval(refresh, 1000);
</script>
</body>
</html>
"""


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def matrix_from_pos_xmat(pos: np.ndarray, xmat: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(xmat, dtype=np.float64).reshape(3, 3)
    matrix[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    return matrix


def save_pose(path: Path, matrix: np.ndarray) -> None:
    rpy = Rotation.from_matrix(matrix[:3, :3]).as_euler("xyz")
    payload = {
        "x": float(matrix[0, 3]),
        "y": float(matrix[1, 3]),
        "z": float(matrix[2, 3]),
        "rx": float(rpy[0]),
        "ry": float(rpy[1]),
        "rz": float(rpy[2]),
        "position_unit": "m",
        "angle_unit": "rad",
    }
    write_json(path, payload)


def save_matrix(path: Path, matrix: np.ndarray, transform_name: str) -> None:
    write_json(path, {
        "transform": transform_name,
        "matrix_4x4": np.asarray(matrix, dtype=np.float64).tolist(),
    })


def camera_world_from_mujoco(model: mujoco.MjModel, data: mujoco.MjData, camera_name: str) -> np.ndarray:
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    camera_pos = data.cam_xpos[camera_id]
    camera_rot_mj = data.cam_xmat[camera_id].reshape(3, 3)
    cv_from_mj = np.diag([1.0, -1.0, -1.0])
    return matrix_from_pos_xmat(camera_pos, camera_rot_mj @ cv_from_mj)


def look_at_xyaxes(pos: np.ndarray, target: np.ndarray) -> str:
    forward = target - pos
    forward /= np.linalg.norm(forward)
    up = np.array([0.0, 0.0, 1.0])
    x_axis = np.cross(forward, up)
    x_axis /= np.linalg.norm(x_axis)
    z_axis = -forward
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    return " ".join(f"{v:.9g}" for v in np.r_[x_axis, y_axis])


def download_file(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=30) as response:
        target.write_bytes(response.read())


def github_contents(path: str) -> list[dict]:
    url = f"https://api.github.com/repos/google-deepmind/mujoco_menagerie/contents/{path}?ref=main"
    with urlopen(url, timeout=30) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, list):
        raise RuntimeError(f"Unexpected GitHub contents response for {path}")
    return value


def download_github_dir(repo_path: str, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for item in github_contents(repo_path):
        name = item["name"]
        kind = item["type"]
        if kind == "dir":
            download_github_dir(item["path"], target / name)
        elif kind == "file":
            download_url = item.get("download_url")
            if not download_url:
                raise RuntimeError(f"GitHub file has no download_url: {item['path']}")
            download_file(download_url, target / name)


def ensure_panda_model(model_root: Path) -> Path:
    panda_root = model_root / PANDA_DIR
    panda_xml = panda_root / "panda.xml"
    if panda_xml.is_file() and (panda_root / "assets").is_dir():
        return panda_xml
    model_root.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Panda model files from {MENAGERIE_REPO}")
    if panda_root.exists():
        for path in sorted(panda_root.rglob("*"), reverse=True):
            path.unlink() if path.is_file() else path.rmdir()
    download_github_dir(PANDA_DIR, panda_root)
    return panda_xml


def chessboard_object_points(columns: int, rows: int, square_size_mm: float) -> np.ndarray:
    square = float(square_size_mm) / 1000.0
    board_width = (columns + 1) * square
    board_height = (rows + 1) * square
    points = np.zeros((columns * rows, 3), dtype=np.float64)
    for row in range(rows):
        for col in range(columns):
            offset = row * columns + col
            points[offset, 0] = -board_width / 2.0 + (col + 1) * square
            points[offset, 1] = board_height / 2.0 - (row + 1) * square
    return points


def append_board_scene(
    panda_xml: Path,
    scene_xml: Path,
    columns: int,
    rows: int,
    square_size_m: float,
    width: int,
    height: int,
) -> None:
    root = ET.parse(panda_xml).getroot()
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "material", {"name": "calib_board_white", "rgba": "1 1 1 1"})
    ET.SubElement(asset, "material", {"name": "calib_board_black", "rgba": "0 0 0 1"})

    hand = root.find(".//body[@name='hand']")
    if hand is None:
        raise RuntimeError("Panda model does not contain body 'hand'")
    board_body = ET.SubElement(hand, "body", {
        "name": "calib_board",
        "pos": "0 0 0.245",
        "quat": "0.70710678 -0.70710678 0 0",
    })
    board_width = (columns + 1) * square_size_m
    board_height = (rows + 1) * square_size_m
    margin = square_size_m * 0.35
    ET.SubElement(board_body, "geom", {
        "name": "calib_board_backing",
        "type": "box",
        "size": f"{board_width / 2 + margin:.9g} {board_height / 2 + margin:.9g} 0.0015",
        "material": "calib_board_white",
        "contype": "0",
        "conaffinity": "0",
    })
    for row in range(rows + 1):
        for col in range(columns + 1):
            if (row + col) % 2 == 0:
                continue
            x = -board_width / 2 + (col + 0.5) * square_size_m
            y = board_height / 2 - (row + 0.5) * square_size_m
            ET.SubElement(board_body, "geom", {
                "name": f"calib_square_r{row}_c{col}",
                "type": "box",
                "pos": f"{x:.9g} {y:.9g} 0.003",
                "size": f"{square_size_m * 0.502:.9g} {square_size_m * 0.502:.9g} 0.0007",
                "material": "calib_board_black",
                "contype": "0",
                "conaffinity": "0",
            })
    world = root.find("worldbody")
    if world is None:
        world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", {"pos": "0 -1.5 2.5", "dir": "0.2 0.4 -1", "directional": "true"})
    ET.SubElement(world, "light", {"pos": "1.5 1.5 2.0", "dir": "-0.4 -0.3 -1", "directional": "true"})
    ET.SubElement(world, "geom", {
        "name": "floor",
        "type": "plane",
        "size": "2 2 0.05",
        "rgba": "0.18 0.20 0.22 1",
    })
    camera_pos = np.array([1.05, -1.15, 0.82], dtype=np.float64)
    camera_target = np.array([0.18, 0.0, 0.45], dtype=np.float64)
    ET.SubElement(world, "camera", {
        "name": "fixed_rgb",
        "pos": " ".join(f"{v:.9g}" for v in camera_pos),
        "xyaxes": look_at_xyaxes(camera_pos, camera_target),
        "fovy": "45",
    })

    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "headlight", {"diffuse": "0.6 0.6 0.6", "ambient": "0.35 0.35 0.35", "specular": "0 0 0"})
    global_visual = visual.find("global")
    if global_visual is None:
        global_visual = ET.SubElement(visual, "global")
    global_visual.set("offwidth", str(int(width)))
    global_visual.set("offheight", str(int(height)))
    ET.ElementTree(root).write(scene_xml, encoding="utf-8", xml_declaration=True)


class SimApp:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.width = int(args.width)
        self.height = int(args.height)
        self.square_size_mm = float(args.square_size_mm)
        self.data_dir = Path(args.output_dir).resolve() if args.output_dir else (
            HERE / "data" / datetime.now().strftime("%Y-%m-%d_%H%M%S")
        )
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir = HERE / "third_party"
        panda_xml = ensure_panda_model(self.assets_dir)
        self.scene_xml = panda_xml.parent / "eye_to_hand_scene.xml"
        append_board_scene(
            panda_xml,
            self.scene_xml,
            args.chessboard_columns,
            args.chessboard_rows,
            self.square_size_mm / 1000.0,
            self.width,
            self.height,
        )
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_xml))
        self.data = mujoco.MjData(self.model)
        self.renderer: mujoco.Renderer | None = None
        self.object_points = chessboard_object_points(
            args.chessboard_columns, args.chessboard_rows, self.square_size_mm
        )
        self.joints = DEFAULT_JOINTS.copy()
        self.lock = threading.RLock()
        self.frame_bgr: np.ndarray | None = None
        self.overlay_bgr: np.ndarray | None = None
        self.jpeg: bytes | None = None
        self.quality = {"valid": False, "reason": "starting"}
        self.sample_count = 0
        self.messages: queue.Queue[str] = queue.Queue()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self._init_data()
        self.intrinsics = self._intrinsics()
        self._write_session_files()

    def _init_data(self) -> None:
        self.data.qpos[:7] = self.joints
        if self.model.nq >= 9:
            self.data.qpos[7:9] = 0.035
        self.data.ctrl[:7] = self.joints
        if self.model.nu >= 9:
            self.data.ctrl[7:9] = 0.035
        mujoco.mj_forward(self.model, self.data)

    def _intrinsics(self) -> dict:
        camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "fixed_rgb")
        fovy = float(self.model.cam_fovy[camera_id])
        fy = 0.5 * self.height / math.tan(math.radians(fovy) / 2.0)
        return {
            "width": self.width,
            "height": self.height,
            "fx": fy,
            "fy": fy,
            "cx": (self.width - 1) / 2.0,
            "cy": (self.height - 1) / 2.0,
            "distortion_model": "none",
            "coeffs": [],
        }

    def _write_session_files(self) -> None:
        write_json(self.data_dir / "intrinsics.json", self.intrinsics)
        write_json(self.data_dir / "session.json", {
            "robot_id": "mujoco_franka_emika_panda",
            "calibration_type": "eye_to_hand",
            "camera_mount": "fixed_in_workspace",
            "target_mount": "rigid_on_flange",
            "robot_pose": "T_base_flange",
            "output_transform": "T_base_camera",
            "target_type": TARGET_TYPE,
            "chessboard_columns": self.args.chessboard_columns,
            "chessboard_rows": self.args.chessboard_rows,
            "square_size_mm": self.square_size_mm,
            "image_source": "mujoco.Renderer fixed camera",
            "mujoco_model_source": MENAGERIE_REPO,
            "mujoco_model_dir": str(self.assets_dir / PANDA_DIR),
            "scene_xml": str(self.scene_xml),
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        })

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
        elif self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def _loop(self) -> None:
        period = 1.0 / max(1.0, float(self.args.fps))
        try:
            self._ensure_renderer()
            while not self.stop_event.is_set():
                start = time.monotonic()
                with self.lock:
                    self.data.ctrl[:7] = self.joints
                    if self.model.nu >= 9:
                        self.data.ctrl[7:9] = 0.035
                    for _ in range(8):
                        mujoco.mj_step(self.model, self.data)
                    rgb = self._render_rgb()
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    quality, overlay = self._analyze(bgr)
                    ok, jpeg = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 88])
                    self.frame_bgr = bgr
                    self.overlay_bgr = overlay
                    self.quality = quality
                    self.jpeg = jpeg.tobytes() if ok else None
                elapsed = time.monotonic() - start
                time.sleep(max(0.0, period - elapsed))
        finally:
            if self.renderer is not None:
                self.renderer.close()
                self.renderer = None

    def _render_rgb(self) -> np.ndarray:
        self._ensure_renderer()
        assert self.renderer is not None
        self.renderer.update_scene(self.data, camera="fixed_rgb")
        return self.renderer.render()

    def _ensure_renderer(self) -> None:
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)

    def _analyze(self, bgr: np.ndarray) -> tuple[dict, np.ndarray]:
        overlay = bgr.copy()
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        found, corners = self._detect_chessboard_multiscale(gray)
        pattern = (self.args.chessboard_columns, self.args.chessboard_rows)
        if not found or corners is None:
            cv2.putText(overlay, "invalid: chessboard_not_found", (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            return {"valid": False, "reason": "chessboard_not_found"}, overlay
        cv2.drawChessboardCorners(overlay, pattern, corners, found)
        image_points = corners.reshape(-1, 2).astype(np.float64)
        ok, rvec, tvec = solve_pnp(self.object_points, image_points, self.intrinsics, cv2.SOLVEPNP_ITERATIVE)
        x, y, width, height = cv2.boundingRect(image_points.astype(np.float32))
        coverage = float(width * height) / float(gray.shape[0] * gray.shape[1])
        if not ok:
            return {"valid": False, "reason": "solve_pnp_failed", "coverage": coverage}, overlay
        rms = reprojection_rms_px(self.object_points, image_points, rvec, tvec, self.intrinsics)
        valid = coverage >= self.args.min_board_coverage and rms <= self.args.max_reprojection_px
        reason = "ok" if valid else ("board_too_small" if coverage < self.args.min_board_coverage else "reprojection_high")
        color = (80, 220, 80) if valid else (0, 210, 255)
        cv2.putText(overlay, f"{reason} rms={rms:.3f}px coverage={coverage:.1%}", (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
        return {
            "valid": bool(valid),
            "reason": reason,
            "corner_count": int(image_points.shape[0]),
            "coverage": coverage,
            "reprojection_rms_px": float(rms),
        }, overlay

    def _detect_chessboard_multiscale(self, gray: np.ndarray):
        pattern = (self.args.chessboard_columns, self.args.chessboard_rows)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        if hasattr(cv2, "CALIB_CB_EXHAUSTIVE"):
            flags |= cv2.CALIB_CB_EXHAUSTIVE
        for scale in (1.0, 0.75, 0.5):
            image = gray if scale == 1.0 else cv2.resize(gray, None, fx=scale, fy=scale)
            found, corners = cv2.findChessboardCorners(image, pattern, flags)
            if not found or corners is None:
                continue
            if scale != 1.0:
                corners = (corners.astype(np.float64) / scale).astype(np.float32)
            corners = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
            )
            return True, corners
        return False, None

    def status(self) -> dict:
        with self.lock:
            return {
                "sample_count": self.sample_count,
                "quality": dict(self.quality),
                "joints": self.joints.tolist(),
                "data_dir": str(self.data_dir),
            }

    def set_joints(self, joints: list[float]) -> list[float]:
        values = np.asarray(joints, dtype=np.float64).reshape(7)
        values = np.clip(values, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        with self.lock:
            self.joints = values
        return values.tolist()

    def random_pose(self) -> list[float]:
        for _ in range(200):
            values = np.array([random.uniform(lo, hi) for lo, hi in JOINT_LIMITS], dtype=np.float64)
            if self._pose_is_reasonable(values):
                return self.set_joints(values.tolist())
        return self.set_joints(DEFAULT_JOINTS.tolist())

    def _pose_is_reasonable(self, joints: np.ndarray) -> bool:
        with self.lock:
            old_qpos = self.data.qpos.copy()
            old_ctrl = self.data.ctrl.copy()
            self.data.qpos[:7] = joints
            self.data.ctrl[:7] = joints
            mujoco.mj_forward(self.model, self.data)
            hand_z = float(self.data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")][2])
            self.data.qpos[:] = old_qpos
            self.data.ctrl[:] = old_ctrl
            mujoco.mj_forward(self.model, self.data)
        return hand_z > 0.18

    def capture(self) -> dict:
        with self.lock:
            if self.frame_bgr is None:
                raise RuntimeError("No frame available yet")
            if not self.quality.get("valid"):
                raise RuntimeError(f"Current frame is invalid: {self.quality.get('reason')}")
            index = self.sample_count
            frame = self.frame_bgr.copy()
            hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
            board_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "calib_board")
            base_flange = matrix_from_pos_xmat(self.data.xpos[hand_id], self.data.xmat[hand_id])
            base_camera = camera_world_from_mujoco(self.model, self.data, "fixed_rgb")
            base_target = matrix_from_pos_xmat(self.data.xpos[board_id], self.data.xmat[board_id])
            camera_target = np.linalg.inv(base_camera) @ base_target
            cv2.imwrite(str(self.data_dir / f"frame_{index:03d}.png"), frame)
            save_pose(self.data_dir / f"pose_{index:03d}.json", base_flange)
            save_matrix(self.data_dir / f"target_{index:03d}.json", camera_target, "T_camera_target")
            save_matrix(self.data_dir / "ground_truth_T_base_camera.json", base_camera, "T_base_camera")
            self.sample_count += 1
            return {"index": index, "message": f"已采集 frame_{index:03d}.png"}

    def solve(self) -> dict:
        with self.lock:
            count = self.sample_count
        if count < 8:
            raise RuntimeError(f"有效样本不足：{count} < 8")
        if self.args.solve_source == "truth":
            result = self.solve_from_truth_targets()
            error = result["ground_truth_error"]
            return {
                "message": (
                    f"求解完成，真值误差 "
                    f"{error['translation_mm']:.3f} mm / {error['rotation_deg']:.3f} deg"
                )
            }
        result = self.solve_from_images()
        gt_path = self.data_dir / "ground_truth_T_base_camera.json"
        if gt_path.is_file():
            gt = np.asarray(json.loads(gt_path.read_text(encoding="utf-8"))["matrix_4x4"], dtype=np.float64)
            estimate = np.asarray(result["matrix_4x4"], dtype=np.float64)
            delta = np.linalg.inv(gt) @ estimate
            trans_mm = float(np.linalg.norm(delta[:3, 3]) * 1000.0)
            rot_deg = float(np.degrees(np.linalg.norm(Rotation.from_matrix(delta[:3, :3]).as_rotvec())))
            result["ground_truth_error"] = {
                "translation_mm": trans_mm,
                "rotation_deg": rot_deg,
            }
            write_json(self.data_dir / "T_base_camera.json", result)
            return {"message": f"求解完成，真值误差 {trans_mm:.3f} mm / {rot_deg:.3f} deg"}
        return {"message": "求解完成"}

    def solve_from_images(self) -> dict:
        pairs, skipped = [], []
        for index in range(self.sample_count):
            image_path = self.data_dir / f"frame_{index:03d}.png"
            pose_path = self.data_dir / f"pose_{index:03d}.json"
            if not image_path.is_file() or not pose_path.is_file():
                skipped.append({"index": index, "reason": "missing_image_or_pose"})
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                skipped.append({"index": index, "reason": "image_unreadable"})
                continue
            found, corners = self._detect_chessboard_multiscale(
                cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            )
            if not found or corners is None:
                skipped.append({"index": index, "reason": "chessboard_not_found"})
                continue
            image_points = corners.reshape(-1, 2).astype(np.float64)
            ok, rvec, tvec = solve_pnp(
                self.object_points, image_points, self.intrinsics, cv2.SOLVEPNP_ITERATIVE
            )
            if not ok:
                skipped.append({"index": index, "reason": "solve_pnp_failed"})
                continue
            pairs.append({
                "index": index,
                "base_flange": load_robot_pose(pose_path),
                "camera_target": matrix_from_pos_xmat(tvec, cv2.Rodrigues(rvec)[0]),
                "reprojection_rms_px": reprojection_rms_px(
                    self.object_points, image_points, rvec, tvec, self.intrinsics
                ),
            })
        if len(pairs) < 8:
            raise RuntimeError(f"棋盘格有效样本不足：{len(pairs)} < 8")
        method_name = self.args.method.upper()
        base_camera = solve_matrix(pairs, METHODS[method_name])
        metrics = evaluate_matrix(pairs, base_camera)
        rpy_deg = Rotation.from_matrix(base_camera[:3, :3]).as_euler("xyz", degrees=True)
        result = {
            "schema_version": 1,
            "transform": "T_base_camera",
            "meaning": "OpenCV camera coordinates to MuJoCo world/base coordinates",
            "calibration_type": "eye_to_hand",
            "method": method_name,
            "source": "chessboard_image_pnp",
            "validated": False,
            "source_dir": str(self.data_dir),
            "robot_id": "mujoco_franka_emika_panda",
            "target_type": TARGET_TYPE,
            "chessboard_columns": self.args.chessboard_columns,
            "chessboard_rows": self.args.chessboard_rows,
            "square_size_mm": self.square_size_mm,
            "recorded_sample_count": self.sample_count,
            "valid_sample_count": len(pairs),
            "valid_indices": [item["index"] for item in pairs],
            "skipped": skipped,
            "matrix_4x4": base_camera.tolist(),
            "matrix_4x4_flat": base_camera.flatten().tolist(),
            "xyz_m": base_camera[:3, 3].tolist(),
            "rpy_deg": rpy_deg.tolist(),
            "consistency": metrics,
        }
        write_json(self.data_dir / "T_base_camera.json", result)
        return result

    def solve_from_truth_targets(self) -> dict:
        pairs = []
        for index in range(self.sample_count):
            pose_path = self.data_dir / f"pose_{index:03d}.json"
            target_path = self.data_dir / f"target_{index:03d}.json"
            if not pose_path.is_file() or not target_path.is_file():
                continue
            pairs.append({
                "index": index,
                "base_flange": load_robot_pose(pose_path),
                "camera_target": np.asarray(
                    json.loads(target_path.read_text(encoding="utf-8"))["matrix_4x4"],
                    dtype=np.float64,
                ),
                "reprojection_rms_px": None,
            })
        if len(pairs) < 8:
            raise RuntimeError(f"有效真值样本不足：{len(pairs)} < 8")
        method_name = self.args.method.upper()
        base_camera = solve_matrix(pairs, METHODS[method_name])
        metrics = evaluate_matrix(pairs, base_camera)
        rpy_deg = Rotation.from_matrix(base_camera[:3, :3]).as_euler("xyz", degrees=True)
        gt = np.asarray(
            json.loads((self.data_dir / "ground_truth_T_base_camera.json").read_text(encoding="utf-8"))["matrix_4x4"],
            dtype=np.float64,
        )
        delta = np.linalg.inv(gt) @ base_camera
        trans_mm = float(np.linalg.norm(delta[:3, 3]) * 1000.0)
        rot_deg = float(np.degrees(np.linalg.norm(Rotation.from_matrix(delta[:3, :3]).as_rotvec())))
        result = {
            "schema_version": 1,
            "transform": "T_base_camera",
            "meaning": "OpenCV camera coordinates to MuJoCo world/base coordinates",
            "calibration_type": "eye_to_hand",
            "method": method_name,
            "source": "mujoco_ground_truth_T_camera_target",
            "validated": True,
            "source_dir": str(self.data_dir),
            "robot_id": "mujoco_franka_emika_panda",
            "target_type": TARGET_TYPE,
            "chessboard_columns": self.args.chessboard_columns,
            "chessboard_rows": self.args.chessboard_rows,
            "square_size_mm": self.square_size_mm,
            "valid_sample_count": len(pairs),
            "valid_indices": [item["index"] for item in pairs],
            "matrix_4x4": base_camera.tolist(),
            "matrix_4x4_flat": base_camera.flatten().tolist(),
            "xyz_m": base_camera[:3, 3].tolist(),
            "rpy_deg": rpy_deg.tolist(),
            "consistency": metrics,
            "ground_truth_error": {
                "translation_mm": trans_mm,
                "rotation_deg": rot_deg,
            },
        }
        write_json(self.data_dir / "T_base_camera.json", result)
        print("\nMuJoCo truth-target eye-to-hand result: T_base_camera")
        print(base_camera)
        print(f"Ground-truth error: {trans_mm:.6f} mm, {rot_deg:.6f} deg")
        return result

    def wait_for_jpeg(self) -> bytes:
        while not self.stop_event.is_set():
            with self.lock:
                if self.jpeg is not None:
                    return self.jpeg
            time.sleep(0.02)
        return b""


def create_flask(sim: SimApp) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        return render_template_string(INDEX_HTML, limits=JOINT_LIMITS.tolist())

    @app.get("/api/status")
    def status():
        return jsonify(sim.status())

    @app.post("/api/joints")
    def joints():
        payload = request.get_json(force=True)
        return jsonify({"joints": sim.set_joints(payload["joints"])})

    @app.post("/api/random_pose")
    def random_pose():
        return jsonify({"joints": sim.random_pose()})

    @app.post("/api/home")
    def home():
        return jsonify({"joints": sim.set_joints(DEFAULT_JOINTS.tolist())})

    @app.post("/api/capture")
    def capture():
        try:
            return jsonify(sim.capture())
        except (RuntimeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/solve")
    def solve():
        try:
            return jsonify(sim.solve())
        except (RuntimeError, ValueError, cv2.error) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/stream.mjpg")
    def stream():
        def generate():
            while not sim.stop_event.is_set():
                jpeg = sim.wait_for_jpeg()
                if not jpeg:
                    break
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                time.sleep(0.03)
        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/favicon.ico")
    def favicon():
        return redirect(url_for("static", filename="favicon.ico"))

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MuJoCo eye-to-hand RGB collection simulator")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--output-dir")
    parser.add_argument("--chessboard-columns", type=int, default=7,
                        help="number of inner corners along the board X direction")
    parser.add_argument("--chessboard-rows", type=int, default=6,
                        help="number of inner corners along the board Y direction")
    parser.add_argument("--square-size-mm", type=float, default=40.0)
    parser.add_argument("--min-board-coverage", type=float, default=0.015)
    parser.add_argument("--max-reprojection-px", type=float, default=2.0)
    parser.add_argument("--method", choices=sorted(METHODS), default="PARK")
    parser.add_argument("--solve-source", choices=["truth", "image"], default="truth")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.width <= 0 or args.height <= 0:
        raise SystemExit("--width/--height must be positive")
    if args.chessboard_columns < 3 or args.chessboard_rows < 3:
        raise SystemExit("--chessboard-columns/--chessboard-rows must be at least 3")
    if args.square_size_mm <= 0:
        raise SystemExit("--square-size-mm must be positive")
    if not 0.0 <= args.min_board_coverage <= 1.0:
        raise SystemExit("--min-board-coverage must be between 0 and 1")
    try:
        sim = SimApp(args)
    except (OSError, RuntimeError, URLError, mujoco.FatalError) as exc:
        raise SystemExit(str(exc)) from exc
    sim.start()
    app = create_flask(sim)
    print(f"浏览器采集界面: http://{args.host}:{args.port}")
    print(f"数据保存目录: {sim.data_dir}")
    try:
        app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    finally:
        sim.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
