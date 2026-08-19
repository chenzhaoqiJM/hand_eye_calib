#!/usr/bin/env python3
"""Generate a dimensionally accurate chessboard calibration PDF."""

from __future__ import annotations

import argparse
from pathlib import Path

from reportlab.lib.pagesizes import A4, LETTER, landscape, portrait
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas


PAGE_SIZES = {"A4": A4, "LETTER": LETTER}


def positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


def generate_pdf(
    output: Path,
    columns: int,
    rows: int,
    square_size_mm: float,
    page_size_name: str,
    orientation: str,
    margin_mm: float,
) -> None:
    """Draw a vector chessboard where columns/rows are inner-corner counts."""
    page_size = PAGE_SIZES[page_size_name]
    page_width, page_height = (
        landscape(page_size) if orientation == "landscape" else portrait(page_size)
    )
    square_size = square_size_mm * mm
    square_columns = columns + 1
    square_rows = rows + 1
    board_width = square_columns * square_size
    board_height = square_rows * square_size
    margin = margin_mm * mm
    label_space = 14 * mm
    required_width = board_width + 2 * margin
    required_height = board_height + 2 * margin + label_space
    if required_width > page_width or required_height > page_height:
        available_width_mm = (page_width - 2 * margin) / mm
        available_height_mm = (page_height - 2 * margin - label_space) / mm
        maximum_square_mm = min(
            available_width_mm / square_columns,
            available_height_mm / square_rows,
        )
        raise ValueError(
            f"{square_columns}x{square_rows} squares at {square_size_mm:g} mm do not "
            f"fit on {page_size_name} {orientation}; maximum square size is "
            f"approximately {maximum_square_mm:.2f} mm with current margins"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    left = (page_width - board_width) / 2
    bottom = (page_height - board_height + label_space) / 2
    pdf = canvas.Canvas(str(output), pagesize=(page_width, page_height))
    pdf.setTitle(
        f"Chessboard {columns}x{rows} inner corners - {square_size_mm:g} mm"
    )
    pdf.setAuthor("hand_eye_calib")
    pdf.setFillColorRGB(1, 1, 1)
    pdf.rect(0, 0, page_width, page_height, stroke=0, fill=1)
    pdf.setFillColorRGB(0, 0, 0)
    for row in range(square_rows):
        for column in range(square_columns):
            if (row + column) % 2 == 0:
                pdf.rect(
                    left + column * square_size,
                    bottom + row * square_size,
                    square_size,
                    square_size,
                    stroke=0,
                    fill=1,
                )

    label_y = bottom - 8 * mm
    pdf.setFont("Helvetica", 9)
    pdf.drawCentredString(
        page_width / 2,
        label_y,
        f"Chessboard: {columns} x {rows} inner corners | "
        f"square: {square_size_mm:g} mm | print at 100% / Actual size",
    )
    ruler_width = 100 * mm
    ruler_left = (page_width - ruler_width) / 2
    ruler_y = label_y - 5 * mm
    pdf.setLineWidth(0.4)
    pdf.line(ruler_left, ruler_y, ruler_left + ruler_width, ruler_y)
    for value in range(0, 101, 10):
        x = ruler_left + value * mm
        tick = 2.5 * mm if value % 50 == 0 else 1.5 * mm
        pdf.line(x, ruler_y - tick / 2, x, ruler_y + tick / 2)
    pdf.setFont("Helvetica", 7)
    pdf.drawCentredString(page_width / 2, ruler_y - 4 * mm, "100 mm verification ruler")
    pdf.showPage()
    pdf.save()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a print-ready vector chessboard calibration PDF"
    )
    parser.add_argument("--columns", type=int, default=9, help="inner corner columns")
    parser.add_argument("--rows", type=int, default=6, help="inner corner rows")
    parser.add_argument(
        "--square-size-mm",
        type=positive_float,
        default=25.0,
        help="physical side length of each square",
    )
    parser.add_argument("--page-size", choices=PAGE_SIZES, default="A4")
    parser.add_argument(
        "--orientation", choices=("landscape", "portrait"), default="landscape"
    )
    parser.add_argument(
        "--margin-mm", type=positive_float, default=8.0, help="minimum white margin"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.columns < 3 or args.rows < 3:
        parser.error("--columns and --rows must be at least 3")

    output = args.output or Path(
        f"chessboard_{args.columns}x{args.rows}_{args.square_size_mm:g}mm.pdf"
    )
    try:
        generate_pdf(
            output.resolve(),
            args.columns,
            args.rows,
            args.square_size_mm,
            args.page_size,
            args.orientation,
            args.margin_mm,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    square_columns, square_rows = args.columns + 1, args.rows + 1
    print(f"Generated: {output.resolve()}")
    print(f"Inner corners: {args.columns} x {args.rows}")
    print(f"Squares: {square_columns} x {square_rows}")
    print(f"Square size: {args.square_size_mm:g} x {args.square_size_mm:g} mm")
    print(
        f"Board size: {square_columns * args.square_size_mm:g} x "
        f"{square_rows * args.square_size_mm:g} mm"
    )
    print("Print at 100% / Actual size; disable Fit, Shrink, and Scale to page.")
    print("After printing, verify several squares and the 100 mm ruler with a ruler.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
