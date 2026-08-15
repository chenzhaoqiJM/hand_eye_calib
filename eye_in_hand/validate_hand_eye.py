#!/usr/bin/env python3
"""Compare hand-eye methods and fixed-target consistency from recorded data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from calibrate_from_data import load_robot_pose, tag_points
from camera_model import reprojection_rms_px, solve_pnp


TAG_FAMILY = cv2.aruco.DICT_APRILTAG_36h11
METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = np.asarray(translation).reshape(3)
    return matrix


def recorded_indices(data_dir: Path) -> list[int]:
    pattern = re.compile(r"frame_(\d{3})\.(png|jpg|jpeg|bmp)$", re.IGNORECASE)
    return sorted(
        int(match.group(1))
        for path in data_dir.iterdir()
        if (match := pattern.match(path.name))
    )


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def read_image(path: Path):
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None


def collect_pairs(data_dir: Path, tag_size_mm: float):
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(TAG_FAMILY),
        cv2.aruco.DetectorParameters(),
    )
    intrinsics = read_json(data_dir / "intrinsics.json")
    object_points = tag_points(tag_size_mm)
    pairs = []
    skipped = []
    expected_tag_id = None

    for index in recorded_indices(data_dir):
        image_path = data_dir / f"frame_{index:03d}.png"
        pose_path = data_dir / f"pose_{index:03d}.json"
        image = read_image(image_path)
        if image is None:
            skipped.append({"index": index, "reason": "image_unreadable"})
            continue
        if not pose_path.is_file():
            skipped.append({"index": index, "reason": "pose_missing"})
            continue

        corners, ids, _ = detector.detectMarkers(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        if ids is None or len(corners) != 1:
            skipped.append(
                {
                    "index": index,
                    "reason": "tag_count_not_one",
                    "tag_count": 0 if ids is None else len(corners),
                }
            )
            continue

        tag_id = int(ids.reshape(-1)[0])
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
            continue

        image_points = corners[0].reshape(-1, 2).astype(np.float64)
        ok, rvec, tvec = solve_pnp(
            object_points,
            image_points,
            intrinsics,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok:
            skipped.append({"index": index, "reason": "solve_pnp_failed"})
            continue

        flange_rotation, flange_translation = load_robot_pose(pose_path)
        base_flange = transform(flange_rotation, flange_translation)
        camera_target = transform(cv2.Rodrigues(rvec)[0], tvec)
        pairs.append(
            {
                "index": index,
                "tag_id": tag_id,
                "base_flange": base_flange,
                "camera_target": camera_target,
                "reprojection_rms_px": reprojection_rms_px(
                    object_points, image_points, rvec, tvec, intrinsics
                ),
            }
        )

    return pairs, skipped, expected_tag_id, intrinsics


def solve_matrix(pairs: list[dict], method: int) -> np.ndarray:
    rotation, translation = cv2.calibrateHandEye(
        [item["base_flange"][:3, :3] for item in pairs],
        [item["base_flange"][:3, 3].reshape(3, 1) for item in pairs],
        [item["camera_target"][:3, :3] for item in pairs],
        [item["camera_target"][:3, 3].reshape(3, 1) for item in pairs],
        method=method,
    )
    return transform(rotation, translation)


def evaluate_matrix(pairs: list[dict], flange_camera: np.ndarray) -> dict:
    base_targets = [
        item["base_flange"] @ flange_camera @ item["camera_target"] for item in pairs
    ]
    translations = np.stack([matrix[:3, 3] for matrix in base_targets])
    rotations = Rotation.from_matrix(np.stack([matrix[:3, :3] for matrix in base_targets]))
    mean_translation = np.mean(translations, axis=0)
    mean_rotation = rotations.mean()
    translation_error_m = np.linalg.norm(translations - mean_translation, axis=1)
    rotation_error_deg = np.degrees(
        np.linalg.norm((rotations * mean_rotation.inv()).as_rotvec(), axis=1)
    )
    return {
        "translation_rms_mm": float(np.sqrt(np.mean(translation_error_m ** 2)) * 1000.0),
        "translation_median_mm": float(np.median(translation_error_m) * 1000.0),
        "translation_p95_mm": float(np.percentile(translation_error_m, 95) * 1000.0),
        "translation_max_mm": float(np.max(translation_error_m) * 1000.0),
        "rotation_rms_deg": float(np.sqrt(np.mean(rotation_error_deg ** 2))),
        "rotation_median_deg": float(np.median(rotation_error_deg)),
        "rotation_p95_deg": float(np.percentile(rotation_error_deg, 95)),
        "rotation_max_deg": float(np.max(rotation_error_deg)),
        "fixed_target_mean_base_m": mean_translation.tolist(),
        "per_sample": [
            {
                "index": item["index"],
                "translation_error_mm": float(translation_error_m[offset] * 1000.0),
                "rotation_error_deg": float(rotation_error_deg[offset]),
                "pnp_reprojection_rms_px": item["reprojection_rms_px"],
            }
            for offset, item in enumerate(pairs)
        ],
    }


def load_config_matrix(config_path: Path) -> np.ndarray:
    import yaml

    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    return np.asarray(config["hand_eye"]["matrix_4x4"], dtype=np.float64)


def matrix_delta(left: np.ndarray, right: np.ndarray) -> dict:
    delta = np.linalg.inv(left) @ right
    return {
        "translation_mm": float(np.linalg.norm(delta[:3, 3]) * 1000.0),
        "rotation_deg": float(
            np.degrees(np.linalg.norm(Rotation.from_matrix(delta[:3, :3]).as_rotvec()))
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate hand-eye candidates with fixed AprilTag consistency"
    )
    parser.add_argument("data_dir")
    parser.add_argument("--tag-size-mm", type=float, default=50.0)
    parser.add_argument("--config", default=None, help="Also evaluate a config.yaml matrix")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    pairs, skipped, tag_id, intrinsics = collect_pairs(data_dir, args.tag_size_mm)
    if len(pairs) < 8:
        raise SystemExit(f"Not enough valid pairs: {len(pairs)} < 8")

    candidates = {}
    for name, method in METHODS.items():
        try:
            matrix = solve_matrix(pairs, method)
            metrics = evaluate_matrix(pairs, matrix)
            metrics["matrix_4x4"] = matrix.tolist()
            metrics["score"] = (
                metrics["translation_rms_mm"] + 2.0 * metrics["rotation_rms_deg"]
            )
            candidates[name] = metrics
        except cv2.error as exc:
            candidates[name] = {"error": str(exc)}

    valid = {name: value for name, value in candidates.items() if "score" in value}
    best_name = min(valid, key=lambda name: valid[name]["score"])
    best_matrix = np.asarray(valid[best_name]["matrix_4x4"])

    report = {
        "schema_version": 1,
        "frame_convention": "T_base_target=T_base_flange@T_flange_camera@T_camera_target",
        "data_dir": str(data_dir),
        "tag_size_mm": args.tag_size_mm,
        "tag_id": tag_id,
        "valid_sample_count": len(pairs),
        "skipped": skipped,
        "pnp_distortion_model": intrinsics.get(
            "distortion_model", "legacy_missing_assumed_zero"
        ),
        "best_method": best_name,
        "candidates": candidates,
    }
    if args.config:
        current = load_config_matrix(Path(args.config).resolve())
        report["current_config"] = evaluate_matrix(pairs, current)
        report["current_config"]["matrix_4x4"] = current.tolist()
        report["current_to_best_delta"] = matrix_delta(current, best_matrix)

    output = (
        Path(args.output).resolve()
        if args.output
        else data_dir / "hand_eye_validation.json"
    )
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Valid samples: {len(pairs)}, skipped: {len(skipped)}")
    for name, value in candidates.items():
        if "score" not in value:
            print(f"{name:10s} FAILED")
            continue
        print(
            f"{name:10s} translation_rms={value['translation_rms_mm']:.2f} mm "
            f"rotation_rms={value['rotation_rms_deg']:.2f} deg"
        )
    print(f"Best candidate from this dataset: {best_name}")
    if "current_to_best_delta" in report:
        delta = report["current_to_best_delta"]
        print(
            f"Current config delta to best: {delta['translation_mm']:.2f} mm, "
            f"{delta['rotation_deg']:.2f} deg"
        )
    print(f"Report saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
