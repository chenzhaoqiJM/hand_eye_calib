#!/usr/bin/env python3
"""Compare eye-to-hand methods using fixed flange-to-target consistency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from calibrate_from_data import METHODS, collect_pairs, evaluate_matrix, solve_matrix


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate eye-to-hand calibration methods")
    parser.add_argument("data_dir")
    parser.add_argument("--tag-size-mm", type=float, default=None)
    parser.add_argument("--output")
    args = parser.parse_args()
    data_dir = Path(args.data_dir).resolve()
    try:
        session_path = data_dir / "session.json"
        session = json.loads(session_path.read_text(encoding="utf-8")) if session_path.is_file() else {}
        size = args.tag_size_mm if args.tag_size_mm is not None else float(session.get("tag_size_mm", 50.0))
        pairs, skipped, tag_id, _, _ = collect_pairs(data_dir, size)
        if len(pairs) < 8:
            raise RuntimeError(f"Not enough valid pairs: {len(pairs)} < 8")
        candidates = {}
        for name, method in METHODS.items():
            try:
                matrix = solve_matrix(pairs, method)
                metrics = evaluate_matrix(pairs, matrix)
                metrics["matrix_4x4"] = matrix.tolist()
                metrics["score"] = metrics["translation_rms_mm"] + 2.0 * metrics["rotation_rms_deg"]
                candidates[name] = metrics
            except cv2.error as exc:
                candidates[name] = {"error": str(exc)}
        valid = {name: value for name, value in candidates.items() if "score" in value}
        if not valid:
            raise RuntimeError("All eye-to-hand methods failed")
        best = min(valid, key=lambda name: valid[name]["score"])
        report = {
            "schema_version": 1,
            "frame_convention": "T_flange_target=inv(T_base_flange)@T_base_camera@T_camera_target",
            "data_dir": str(data_dir),
            "tag_size_mm": size,
            "tag_id": tag_id,
            "valid_sample_count": len(pairs),
            "skipped": skipped,
            "best_method": best,
            "candidates": candidates,
        }
        output = Path(args.output).resolve() if args.output else data_dir / "eye_to_hand_validation.json"
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        for name, value in candidates.items():
            if "score" not in value:
                print(f"{name:10s} FAILED")
            else:
                print(f"{name:10s} translation_rms={value['translation_rms_mm']:.3f} mm "
                      f"rotation_rms={value['rotation_rms_deg']:.3f} deg")
        print(f"Best method: {best}")
        print(f"Saved report: {output}")
    except (OSError, ValueError, RuntimeError, cv2.error) as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())