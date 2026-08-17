#!/usr/bin/env python3
"""Solve eye-to-hand calibration from RGB images and flange poses.

Robot poses are T_base_flange. The AprilTag is rigidly attached to the flange.
PnP estimates T_camera_target. The output is T_base_camera (camera -> base).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from camera_model import reprojection_rms_px, solve_pnp, validate_intrinsics


DEFAULT_TAG_SIZE_MM = 50.0
MIN_VALID_PAIRS = 8
TAG_FAMILY = cv2.aruco.DICT_APRILTAG_36h11
TAG_FAMILY_NAME = "DICT_APRILTAG_36h11"
METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(rotation, dtype=np.float64)
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return matrix


def invert_transform(matrix: np.ndarray) -> np.ndarray:
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def tag_points(size_mm: float) -> np.ndarray:
    if not np.isfinite(size_mm) or size_mm <= 0:
        raise ValueError("tag_size_mm must be positive")
    half = float(size_mm) / 2000.0
    return np.array(
        [[-half, half, 0.0], [half, half, 0.0],
         [half, -half, 0.0], [-half, -half, 0.0]],
        dtype=np.float64,
    )


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Top-level JSON value must be an object: {path}")
    return value


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize_pose_units(values, position_unit: str, angle_unit: str) -> np.ndarray:
    normalized = np.asarray(values, dtype=np.float64).reshape(6).copy()
    if not np.all(np.isfinite(normalized)):
        raise ValueError("pose values must be finite")
    if position_unit == "mm":
        normalized[:3] /= 1000.0
    elif position_unit != "m":
        raise ValueError("position_unit must be 'm' or 'mm'")
    if angle_unit == "deg":
        normalized[3:] = np.radians(normalized[3:])
    elif angle_unit != "rad":
        raise ValueError("angle_unit must be 'rad' or 'deg'")
    return normalized


def pose_to_matrix(values: np.ndarray) -> np.ndarray:
    return transform(Rotation.from_euler("xyz", values[3:]).as_matrix(), values[:3])


def load_robot_pose(path: Path) -> np.ndarray:
    pose = load_json(path)
    try:
        values = [pose[key] for key in ("x", "y", "z", "rx", "ry", "rz")]
    except KeyError as exc:
        raise ValueError(f"Missing pose field {exc} in {path}") from exc
    normalized = normalize_pose_units(
        values,
        str(pose.get("position_unit", "m")).lower(),
        str(pose.get("angle_unit", "rad")).lower(),
    )
    return pose_to_matrix(normalized)


def import_pose_csv(
    csv_path: str | Path,
    data_dir: str | Path,
    position_unit: str = "m",
    angle_unit: str = "rad",
) -> list[int]:
    fields_required = ("x", "y", "z", "rx", "ry", "rz")
    imported = []
    with Path(csv_path).resolve().open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = {} if reader.fieldnames is None else {
            name.strip().lower(): name for name in reader.fieldnames
        }
        index_key = fields.get("index") or fields.get("idx")
        if index_key is None or any(name not in fields for name in fields_required):
            raise ValueError("CSV columns must include index,x,y,z,rx,ry,rz")
        for row in reader:
            index = int(row[index_key])
            values = normalize_pose_units(
                [row[fields[name]] for name in fields_required], position_unit, angle_unit
            )
            payload = dict(zip(fields_required, map(float, values), strict=True))
            payload.update({"position_unit": "m", "angle_unit": "rad"})
            write_json(Path(data_dir).resolve() / f"pose_{index:03d}.json", payload)
            imported.append(index)
    return imported


def read_image(path: Path):
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None


def recorded_indices(data_dir: Path) -> list[int]:
    pattern = re.compile(r"frame_(\d{3})\.(png|jpg|jpeg|bmp)$", re.IGNORECASE)
    return sorted(
        int(match.group(1)) for path in data_dir.iterdir()
        if (match := pattern.fullmatch(path.name))
    )


def recorded_image_path(data_dir: Path, index: int) -> Path | None:
    for suffix in ("png", "jpg", "jpeg", "bmp"):
        path = data_dir / f"frame_{index:03d}.{suffix}"
        if path.is_file():
            return path
    return None


def detect_target_pose(image, detector, intrinsics: dict, object_points: np.ndarray):
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    count = 0 if ids is None else int(np.asarray(ids).size)
    if ids is None or count != 1 or len(corners) != 1:
        return None, {"reason": "tag_count_not_one", "tag_count": count}
    image_points = corners[0].reshape(-1, 2).astype(np.float64)
    ok, rvec, tvec = solve_pnp(
        object_points, image_points, intrinsics, cv2.SOLVEPNP_IPPE_SQUARE
    )
    if not ok:
        return None, {"reason": "solve_pnp_failed"}
    return {
        "tag_id": int(np.asarray(ids).reshape(-1)[0]),
        "camera_target": transform(cv2.Rodrigues(rvec)[0], tvec),
        "reprojection_rms_px": reprojection_rms_px(
            object_points, image_points, rvec, tvec, intrinsics
        ),
    }, None


def collect_pairs(data_dir: Path, tag_size_mm: float):
    intrinsics = load_json(data_dir / "intrinsics.json")
    validate_intrinsics(intrinsics)
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(TAG_FAMILY), cv2.aruco.DetectorParameters()
    )
    points = tag_points(tag_size_mm)
    pairs, skipped = [], []
    expected_tag_id = None
    indices = recorded_indices(data_dir)
    for index in indices:
        image_path = recorded_image_path(data_dir, index)
        image = None if image_path is None else read_image(image_path)
        pose_path = data_dir / f"pose_{index:03d}.json"
        if image is None:
            skipped.append({"index": index, "reason": "image_unreadable"})
            continue
        if not pose_path.is_file():
            skipped.append({"index": index, "reason": "pose_missing"})
            continue
        target, rejection = detect_target_pose(image, detector, intrinsics, points)
        if target is None:
            skipped.append({"index": index, **rejection})
            continue
        tag_id = target["tag_id"]
        if expected_tag_id is None:
            expected_tag_id = tag_id
        if tag_id != expected_tag_id:
            skipped.append({"index": index, "reason": "tag_id_mismatch",
                            "expected": expected_tag_id, "actual": tag_id})
            continue
        try:
            base_flange = load_robot_pose(pose_path)
        except (OSError, ValueError) as exc:
            skipped.append({"index": index, "reason": "pose_invalid", "error": str(exc)})
            continue
        pairs.append({"index": index, "base_flange": base_flange, **target})
    return pairs, skipped, expected_tag_id, intrinsics, indices


def solve_matrix(pairs: list[dict], method: int) -> np.ndarray:
    # OpenCV eye-to-hand formulation: feed T_flange_base instead of T_base_flange.
    flange_base = [invert_transform(item["base_flange"]) for item in pairs]
    rotation, translation = cv2.calibrateHandEye(
        [item[:3, :3] for item in flange_base],
        [item[:3, 3].reshape(3, 1) for item in flange_base],
        [item["camera_target"][:3, :3] for item in pairs],
        [item["camera_target"][:3, 3].reshape(3, 1) for item in pairs],
        method=method,
    )
    return transform(rotation, translation)


def evaluate_matrix(pairs: list[dict], base_camera: np.ndarray) -> dict:
    # T_flange_target must be constant because the tag is rigidly mounted on flange.
    flange_targets = [
        invert_transform(item["base_flange"]) @ base_camera @ item["camera_target"]
        for item in pairs
    ]
    translations = np.stack([matrix[:3, 3] for matrix in flange_targets])
    rotations = Rotation.from_matrix(np.stack([matrix[:3, :3] for matrix in flange_targets]))
    mean_translation = translations.mean(axis=0)
    mean_rotation = rotations.mean()
    translation_error = np.linalg.norm(translations - mean_translation, axis=1)
    rotation_error = np.degrees(
        np.linalg.norm((rotations * mean_rotation.inv()).as_rotvec(), axis=1)
    )
    return {
        "translation_rms_mm": float(np.sqrt(np.mean(translation_error**2)) * 1000.0),
        "translation_max_mm": float(np.max(translation_error) * 1000.0),
        "rotation_rms_deg": float(np.sqrt(np.mean(rotation_error**2))),
        "rotation_max_deg": float(np.max(rotation_error)),
        "fixed_T_flange_target_mean": transform(
            mean_rotation.as_matrix(), mean_translation
        ).tolist(),
        "per_sample": [
            {"index": pair["index"],
             "translation_error_mm": float(translation_error[offset] * 1000.0),
             "rotation_error_deg": float(rotation_error[offset]),
             "pnp_reprojection_rms_px": pair["reprojection_rms_px"]}
            for offset, pair in enumerate(pairs)
        ],
    }


def solve_from_data(
    data_dir: str | Path, tag_size_mm: float | None = None, method_name: str = "PARK"
) -> dict:
    path = Path(data_dir).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {path}")
    session_path = path / "session.json"
    session = load_json(session_path) if session_path.is_file() else {}
    session_size = session.get("tag_size_mm")
    size = float(session_size if tag_size_mm is None and session_size is not None
                 else DEFAULT_TAG_SIZE_MM if tag_size_mm is None else tag_size_mm)
    if session_size is not None and tag_size_mm is not None and not np.isclose(size, session_size):
        raise ValueError("--tag-size-mm does not match session.json")
    method_name = method_name.upper()
    if method_name not in METHODS:
        raise ValueError(f"Unknown method: {method_name}")

    pairs, skipped, tag_id, intrinsics, indices = collect_pairs(path, size)
    for item in pairs:
        print(f"  [{item['index']:03d}] OK tag_id={item['tag_id']} "
              f"pnp_rms={item['reprojection_rms_px']:.3f}px")
    for item in skipped:
        print(f"  [{item['index']:03d}] skipped: {item['reason']}")
    if len(pairs) < MIN_VALID_PAIRS:
        raise RuntimeError(f"Not enough valid pairs: {len(pairs)} < {MIN_VALID_PAIRS}")

    base_camera = solve_matrix(pairs, METHODS[method_name])
    if not np.all(np.isfinite(base_camera)):
        raise RuntimeError("OpenCV returned a non-finite calibration matrix")
    metrics = evaluate_matrix(pairs, base_camera)
    rpy_deg = Rotation.from_matrix(base_camera[:3, :3]).as_euler("xyz", degrees=True)
    result_path = path / "T_base_camera.json"
    result = {
        "schema_version": 1,
        "transform": "T_base_camera",
        "meaning": "camera coordinates to robot base coordinates",
        "calibration_type": "eye_to_hand",
        "method": method_name,
        "validated": False,
        "source_dir": str(path),
        "result_path": str(result_path),
        "robot_id": session.get("robot_id"),
        "tag_family": TAG_FAMILY_NAME,
        "tag_id": tag_id,
        "tag_size_mm": size,
        "recorded_sample_count": len(indices),
        "valid_sample_count": len(pairs),
        "valid_indices": [item["index"] for item in pairs],
        "skipped": skipped,
        "matrix_4x4": base_camera.tolist(),
        "matrix_4x4_flat": base_camera.flatten().tolist(),
        "xyz_m": base_camera[:3, 3].tolist(),
        "rpy_deg": rpy_deg.tolist(),
        "consistency": metrics,
        "warning": "Validate with independent poses before production use.",
    }
    write_json(result_path, result)
    print("\nEye-to-hand candidate: T_base_camera (camera -> base)")
    print(base_camera)
    print(f"XYZ (m): {base_camera[:3, 3].tolist()}")
    print(f"RPY xyz (deg): {rpy_deg.tolist()}")
    print(f"Consistency RMS: {metrics['translation_rms_mm']:.3f} mm, "
          f"{metrics['rotation_rms_deg']:.3f} deg")
    print(f"Saved result: {result_path}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Solve eye-to-hand T_base_camera")
    parser.add_argument("data_dir")
    parser.add_argument("--tag-size-mm", type=float, default=None)
    parser.add_argument("--method", choices=sorted(METHODS), default="PARK")
    parser.add_argument("--poses-csv")
    parser.add_argument("--position-unit", choices=["m", "mm"], default="m")
    parser.add_argument("--angle-unit", choices=["rad", "deg"], default="rad")
    args = parser.parse_args()
    try:
        if args.poses_csv:
            imported = import_pose_csv(
                args.poses_csv, args.data_dir, args.position_unit, args.angle_unit
            )
            print(f"Imported {len(imported)} poses")
        solve_from_data(args.data_dir, args.tag_size_mm, args.method)
    except (FileNotFoundError, OSError, ValueError, RuntimeError, cv2.error) as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())