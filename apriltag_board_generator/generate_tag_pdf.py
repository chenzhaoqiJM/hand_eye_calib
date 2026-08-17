#!/usr/bin/env python3
"""Generate a dimensionally accurate AprilTag 36h11 PDF for printing."""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import cv2
from reportlab.lib.pagesizes import A4, LETTER
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas


TAG_FAMILY_NAME = "AprilTag 36h11"
TAG_ID_COUNT = 587
PAGE_SIZES = {"A4": A4, "LETTER": LETTER}


def positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


def generate_marker(tag_id: int, pixels: int) -> object:
    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_APRILTAG_36h11
    )
    return cv2.aruco.generateImageMarker(
        dictionary,
        tag_id,
        pixels,
        borderBits=1,
    )


def generate_pdf(
    output: Path,
    tag_id: int,
    tag_size_mm: float,
    margin_mm: float,
    pixels: int,
    page_size_name: str,
) -> None:
    marker = generate_marker(tag_id, pixels)
    page_width, page_height = PAGE_SIZES[page_size_name]
    tag_size = tag_size_mm * mm
    margin = margin_mm * mm
    quiet_size = tag_size + 2.0 * margin
    if quiet_size > page_width or quiet_size > page_height:
        raise ValueError(
            f"tag plus margins ({tag_size_mm + 2 * margin_mm:g} mm) "
            f"does not fit on {page_size_name}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    left = (page_width - tag_size) / 2.0
    bottom = (page_height - tag_size) / 2.0

    with tempfile.NamedTemporaryFile(suffix=".png") as temporary:
        if not cv2.imwrite(temporary.name, marker):
            raise RuntimeError("failed to write temporary marker image")

        pdf = canvas.Canvas(str(output), pagesize=(page_width, page_height))
        pdf.setTitle(f"{TAG_FAMILY_NAME} ID {tag_id} - {tag_size_mm:g} mm")
        pdf.setAuthor("hand_eye_calib")

        pdf.setFillColorRGB(1, 1, 1)
        pdf.rect(
            left - margin,
            bottom - margin,
            quiet_size,
            quiet_size,
            stroke=0,
            fill=1,
        )
        pdf.drawImage(
            temporary.name,
            left,
            bottom,
            width=tag_size,
            height=tag_size,
            preserveAspectRatio=True,
            mask=None,
        )

        pdf.setFillColorRGB(0, 0, 0)
        pdf.setFont("Helvetica", 9)
        pdf.drawCentredString(
            page_width / 2.0,
            bottom - margin - 8 * mm,
            f"{TAG_FAMILY_NAME} | ID {tag_id} | black square {tag_size_mm:g} mm",
        )
        pdf.showPage()
        pdf.save()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a print-ready AprilTag 36h11 PDF"
    )
    parser.add_argument("--id", type=int, default=0, help="Tag ID: 0 to 586")
    parser.add_argument(
        "--size-mm",
        type=positive_float,
        default=50.0,
        help="Outer black-square side length in millimetres",
    )
    parser.add_argument(
        "--margin-mm",
        type=positive_float,
        default=15.0,
        help="White quiet-zone width around the black square",
    )
    parser.add_argument(
        "--pixels",
        type=int,
        default=1200,
        help="Raster resolution used inside the PDF",
    )
    parser.add_argument("--page-size", choices=PAGE_SIZES, default="A4")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if not 0 <= args.id < TAG_ID_COUNT:
        parser.error("--id must be between 0 and 586")
    if args.pixels < 80:
        parser.error("--pixels must be at least 80")

    output = args.output or Path(
        f"apriltag_36h11_id{args.id}_{args.size_mm:g}mm.pdf"
    )
    try:
        generate_pdf(
            output.resolve(),
            args.id,
            args.size_mm,
            args.margin_mm,
            args.pixels,
            args.page_size,
        )
    except (OSError, RuntimeError, ValueError, cv2.error) as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Generated: {output.resolve()}")
    print(f"Family: {TAG_FAMILY_NAME}")
    print(f"ID: {args.id}")
    print(f"Outer black-square size: {args.size_mm:g} x {args.size_mm:g} mm")
    print(f"White quiet zone: {args.margin_mm:g} mm per side")
    print("Print at 100% / Actual size, then measure the outer black square.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())