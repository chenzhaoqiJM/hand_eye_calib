#!/usr/bin/env python3
"""Solve eye-in-hand calibration from recorded images and manual robot poses.

Expected data layout:
  intrinsics.json
  frame_000.png
  pose_000.json
  frame_001.png
  pose_001.json
  ...

Each pose JSON stores flange pose in base coordinates:
  {"x": 0.1, "y": 0.2, "z": 0.3, "rx": 0.0, "ry": 0.0, "rz": 0.0}

Positions are meters. Rotations are XYZ Euler angles in radians.
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

from camera_model import solve_pnp


DEFAULT_TAG_SIZE_MM = 50.0
MIN_VALID_PAIRS = 8
TAG_FAMILY = cv2.aruco.DICT_APRILTAG_36h11
TAG_FAMILY_NAME = "DICT_APRILTAG_36h11"


def tag_points(size_mm: float) -> np.ndarray:
    if not np.isfinite(size_mm) or float(size_mm) <= 0:
        raise ValueError("tag_size_mm must be positive")
    half = float(size_mm) / 2000.0
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float32,
    )


def rpy_to_rot(rx: float, ry: float, rz: float) -> np.ndarray:
    return Rotation.from_euler("xyz", [rx, ry, rz]).as_matrix()


def rot_to_rpy(rotation: np.ndarray) -> tuple[float, float, float]:
    rpy = Rotation.from_matrix(rotation).as_euler("xyz")
    return float(rpy[0]), float(rpy[1]), float(rpy[2])


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Top-level JSON value must be an object: {path}")
    return value


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def read_image(path: Path):
    """Read through bytes so Unicode paths work on Windows."""
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None


def recorded_indices(data_dir: Path) -> list[int]:
    pattern = re.compile(r"frame_(\d{3})\.(png|jpg|jpeg|bmp)$", re.IGNORECASE)
    return sorted(
        int(match.group(1))
        for path in data_dir.iterdir()
        if (match := pattern.match(path.name))
    )


def load_session(data_dir: Path) -> dict | None:
    path = data_dir / "session.json"
    return load_json(path) if path.is_file() else None


def resolve_tag_size(session: dict | None, requested_mm: float | None) -> float:
    session_size = None if session is None else session.get("tag_size_mm")
    if requested_mm is None:
        size = DEFAULT_TAG_SIZE_MM if session_size is None else float(session_size)
    else:
        size = float(requested_mm)
        if session_size is not None and not np.isclose(size, float(session_size)):
            raise ValueError(
                f"--tag-size-mm={size:g} does not match session.json tag_size_mm={session_size}"
            )
    if not np.isfinite(size) or size <= 0:
        raise ValueError("tag_size_mm must be positive")
    return size


def normalize_pose_units(
    values: np.ndarray,
    position_unit: str = "m",
    angle_unit: str = "rad",
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(6)
    if not np.all(np.isfinite(values)):
        raise ValueError("pose values must be finite")

    if position_unit == "m":
        pass
    elif position_unit == "mm":
        values[:3] /= 1000.0
    else:
        raise ValueError("position_unit must be 'm' or 'mm'")

    if angle_unit == "rad":
        pass
    elif angle_unit == "deg":
        values[3:] = np.radians(values[3:])
    else:
        raise ValueError("angle_unit must be 'rad' or 'deg'")
    return values


def load_robot_pose(path: Path) -> tuple[np.ndarray, np.ndarray]:
    pose = load_json(path)
    try:
        values = np.asarray(
            [pose[key] for key in ("x", "y", "z", "rx", "ry", "rz")],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid pose fields in {path}: {exc}") from exc
    values = normalize_pose_units(
        values,
        str(pose.get("position_unit", "m")).lower(),
        str(pose.get("angle_unit", "rad")).lower(),
    )
    return rpy_to_rot(*values[3:6]).astype(np.float64), values[:3].reshape(3, 1)


def import_pose_csv(
    csv_path: str | Path,
    data_dir: str | Path | None = None,
    position_unit: str = "m",
    angle_unit: str = "rad",
) -> list[int]:
    """Convert a manual CSV pose table to pose_NNN.json files.

    CSV columns: index,x,y,z,rx,ry,rz
    The index column may also be named idx.
    """
    csv_path = Path(csv_path).resolve()
    target_dir = Path(data_dir).resolve() if data_dir else csv_path.parent
    imported: list[int] = []

    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        fields = {name.strip().lower(): name for name in reader.fieldnames}
        index_key = fields.get("index") or fields.get("idx")
        required = [index_key, fields.get("x"), fields.get("y"), fields.get("z"),
                    fields.get("rx"), fields.get("ry"), fields.get("rz")]
        if any(name is None for name in required):
            raise ValueError("CSV columns must include index,x,y,z,rx,ry,rz")

        for row in reader:
            index = int(row[index_key])
            raw = np.asarray(
                [row[fields[name]] for name in ("x", "y", "z", "rx", "ry", "rz")],
                dtype=np.float64,
            )
            values = normalize_pose_units(raw, position_unit, angle_unit)
            pose = {
                "x": float(values[0]),
                "y": float(values[1]),
                "z": float(values[2]),
                "rx": float(values[3]),
                "ry": float(values[4]),
                "rz": float(values[5]),
                "position_unit": "m",
                "angle_unit": "rad",
            }
            write_json(target_dir / f"pose_{index:03d}.json", pose)
            imported.append(index)

    return imported


def detect_target_pose(image, detector, intrinsics: dict, object_points: np.ndarray):
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    tag_count = 0 if ids is None else int(np.asarray(ids).size)
    if ids is None or len(corners) != 1 or tag_count != 1:
        return None, {"reason": "tag_count_not_one", "tag_count": tag_count}

    image_points = corners[0].reshape(-1, 2)
    if image_points.shape != (4, 2):
        return None, {"reason": "tag_corner_count_not_four"}

    ok, rvec, tvec = solve_pnp(
        object_points,
        image_points.astype(np.float64),
        intrinsics,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not ok:
        return None, {"reason": "solve_pnp_failed"}
    rotation, _ = cv2.Rodrigues(rvec)
    return {
        "tag_id": int(np.asarray(ids).reshape(-1)[0]),
        "rotation": rotation.astype(np.float64),
        "translation": np.asarray(tvec, dtype=np.float64).reshape(3, 1),
    }, None


def solve_from_data(data_dir: str | Path, tag_size_mm: float | None = None) -> dict:
    data_dir = Path(data_dir).resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    intrinsics_path = data_dir / "intrinsics.json"
    if not intrinsics_path.is_file():
        raise FileNotFoundError(f"Missing intrinsics.json: {intrinsics_path}")

    session = load_session(data_dir)
    actual_tag_size_mm = resolve_tag_size(session, tag_size_mm)
    robot_id = None if session is None else session.get("robot_id")
    intrinsics = load_json(intrinsics_path)
    object_points = tag_points(actual_tag_size_mm)
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(TAG_FAMILY),
        cv2.aruco.DetectorParameters(),
    )

    r_gripper2base: list[np.ndarray] = []
    t_gripper2base: list[np.ndarray] = []
    r_target2cam: list[np.ndarray] = []
    t_target2cam: list[np.ndarray] = []
    valid: list[int] = []
    skipped: list[dict] = []
    expected_tag_id = None

    indices = recorded_indices(data_dir)
    print(f"Data directory: {data_dir}")
    print("Detecting one consistent AprilTag and pairing it with manual poses:")
    for index in indices:
        image_path = data_dir / f"frame_{index:03d}.png"
        pose_path = data_dir / f"pose_{index:03d}.json"
        image = read_image(image_path)
        if image is None:
            skipped.append({"index": index, "reason": "image_unreadable"})
            print(f"  [{index:03d}] skipped: image_unreadable")
            continue
        if not pose_path.is_file():
            skipped.append({"index": index, "reason": "pose_missing"})
            print(f"  [{index:03d}] skipped: pose_missing")
            continue

        target, rejection = detect_target_pose(
            image, detector, intrinsics, object_points
        )
        if target is None:
            skipped.append({"index": index, **(rejection or {"reason": "tag_rejected"})})
            print(f"  [{index:03d}] skipped: {skipped[-1]['reason']}")
            continue

        tag_id = int(target["tag_id"])
        if expected_tag_id is None:
            expected_tag_id = tag_id
        if tag_id != expected_tag_id:
            skipped.append(
                {
                    "index": index,
                    "reason": "tag_id_mismatch",
                    "expected": expected_tag_id,
                    "actual": tag_id,
                }
            )
            print(f"  [{index:03d}] skipped: tag_id {tag_id} != {expected_tag_id}")
            continue

        try:
            flange_rotation, flange_translation = load_robot_pose(pose_path)
        except (OSError, ValueError) as exc:
            skipped.append({"index": index, "reason": "pose_invalid", "error": str(exc)})
            print(f"  [{index:03d}] skipped: pose_invalid: {exc}")
            continue

        r_gripper2base.append(flange_rotation)
        t_gripper2base.append(flange_translation)
        r_target2cam.append(target["rotation"])
        t_target2cam.append(target["translation"])
        valid.append(index)
        print(f"  [{index:03d}] OK tag_id={tag_id}")

    if len(valid) < MIN_VALID_PAIRS:
        raise RuntimeError(
            f"Not enough valid pairs: {len(valid)} < {MIN_VALID_PAIRS}. "
            "Collect more poses with clear tag detections."
        )

    r_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        r_gripper2base,
        t_gripper2base,
        r_target2cam,
        t_target2cam,
        method=cv2.CALIB_HAND_EYE_PARK,
    )
    r_cam2gripper = np.asarray(r_cam2gripper, dtype=np.float64)
    t_cam2gripper = np.asarray(t_cam2gripper, dtype=np.float64).reshape(3)
    if (
        r_cam2gripper.shape != (3, 3)
        or not np.all(np.isfinite(r_cam2gripper))
        or not np.all(np.isfinite(t_cam2gripper))
    ):
        raise RuntimeError("OpenCV returned an invalid hand-eye matrix")

    orthogonal_error = float(
        np.linalg.norm(r_cam2gripper.T @ r_cam2gripper - np.eye(3))
    )
    determinant = float(np.linalg.det(r_cam2gripper))
    if orthogonal_error > 1e-3 or abs(determinant - 1.0) > 1e-3:
        raise RuntimeError(
            "OpenCV returned a non-rigid rotation: "
            f"orthogonal_error={orthogonal_error:.6g}, det={determinant:.6g}"
        )

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = r_cam2gripper
    transform[:3, 3] = t_cam2gripper
    rpy = rot_to_rpy(r_cam2gripper)
    camera_z_in_flange = r_cam2gripper[:, 2].copy()

    result_path = data_dir / "T_flange_camera.json"
    result = {
        "schema_version": 2,
        "transform": "T_flange_camera",
        "calibration_type": "eye_in_hand",
        "method": "PARK",
        "validated": False,
        "source_dir": str(data_dir),
        "result_path": str(result_path),
        "robot_id": robot_id,
        "tag_family": TAG_FAMILY_NAME,
        "tag_id": expected_tag_id,
        "tag_size_mm": actual_tag_size_mm,
        "pnp_distortion_model": intrinsics.get(
            "distortion_model", "legacy_missing_assumed_zero"
        ),
        "recorded_sample_count": len(indices),
        "valid_sample_count": len(valid),
        "valid_indices": valid,
        "skipped": skipped,
        "skipped_indices": [item["index"] for item in skipped],
        "matrix_4x4": transform.tolist(),
        "matrix_4x4_flat": transform.flatten().tolist(),
        "xyz_m": t_cam2gripper.tolist(),
        "rpy_deg": [float(np.degrees(value)) for value in rpy],
        "camera_z_in_flange": camera_z_in_flange.tolist(),
        "abs_camera_z_dot_flange_z": float(abs(camera_z_in_flange[2])),
        "warning": (
            "This is an unvalidated PARK candidate. Validate with independent "
            "poses before using it in production."
        ),
    }
    write_json(result_path, result)

    pairs_path = data_dir / "valid_pairs.txt"
    with pairs_path.open("w", encoding="utf-8") as stream:
        stream.write("# valid frame/pose indices used by PARK hand-eye calibration\n")
        for index in valid:
            stream.write(
                f"{index:03d} frame_{index:03d}.png pose_{index:03d}.json "
                f"tag_id={expected_tag_id}\n"
            )

    print("\n" + "=" * 60)
    print("Hand-eye candidate: T_flange_camera (camera -> flange)")
    print("=" * 60)
    print(f"Valid pairs: {len(valid)} / {len(indices)}")
    print(f"Valid indices: {valid}")
    print(f"Tag: {TAG_FAMILY_NAME} id={expected_tag_id} size={actual_tag_size_mm:g} mm")
    print(f"\n4x4:\n{transform}")
    print(f"\nXYZ (m): {t_cam2gripper}")
    print(f"RPY (deg): {[float(np.degrees(value)) for value in rpy]}")
    print("\nYAML candidate:")
    print("hand_eye:")
    print("  transform: T_flange_camera")
    print("  calibration_type: eye_in_hand")
    print(f"  robot_id: {json.dumps(robot_id)}")
    print(f"  provenance: {json.dumps(str(result_path))}")
    print("  validated: false")
    print(f"  matrix_4x4: {json.dumps(transform.tolist())}")
    print(f"\nSaved result: {result_path}")
    print(f"Saved valid pair list: {pairs_path}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Solve T_flange_camera from recorded frame_NNN.png + pose_NNN.json"
    )
    parser.add_argument("data_dir", help="Calibration data directory")
    parser.add_argument(
        "--tag-size-mm",
        type=float,
        default=None,
        help="Default: session.json tag_size_mm, otherwise 50",
    )
    parser.add_argument("--poses-csv", default=None, help="Import manual poses first")
    parser.add_argument(
        "--position-unit",
        choices=["m", "mm"],
        default="m",
        help="Unit used by --poses-csv position columns",
    )
    parser.add_argument(
        "--angle-unit",
        choices=["rad", "deg"],
        default="rad",
        help="Unit used by --poses-csv rotation columns",
    )
    args = parser.parse_args()

    try:
        if args.poses_csv:
            imported = import_pose_csv(
                args.poses_csv,
                args.data_dir,
                position_unit=args.position_unit,
                angle_unit=args.angle_unit,
            )
            print(f"Imported {len(imported)} poses from {args.poses_csv}")
        solve_from_data(args.data_dir, args.tag_size_mm)
    except (FileNotFoundError, OSError, ValueError, RuntimeError, cv2.error) as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
