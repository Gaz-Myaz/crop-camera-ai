"""
generate_checkerboard.py

Generates a printable checkerboard calibration target for stereo_calibrate.py.
Run once, print the result, and use it with stereo_capture.py to collect
calibration image pairs.

--------------------------------------------------------------------
Usage:

    uv run generate_checkerboard.py

Produces 'checkerboard.png'. PRINT IT AT 100% / "ACTUAL SIZE" -- do NOT
let your printer or PDF viewer "fit to page" or "scale to fit", or the
square size will be wrong and every distance the calibration produces
will be wrong by the same proportion. Landscape orientation, plain A4 or
US Letter paper both have enough margin at the default settings below.

After printing, measure a few squares with a ruler and confirm they're
close to the nominal size printed in the image footer -- if printer
scaling shifted it, pass the REAL measured size to stereo_calibrate.py's
--square-size-mm instead of trusting the nominal one.

Mount the print on something rigid (cardboard, a clipboard, a book) --
a floppy sheet of paper won't calibrate well, it needs to stay flat.
--------------------------------------------------------------------
"""

import argparse

import cv2
import numpy as np

# 9x6 INTERNAL corners is the OpenCV default pattern size and what
# stereo_calibrate.py expects -- that's a 10x7 grid of squares.
SQUARES_X = 10
SQUARES_Y = 7


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--square-mm", type=float, default=20.0,
                         help="Nominal size of each square in millimeters (default 20mm -- "
                              "chosen so the whole pattern fits comfortably on A4/Letter).")
    parser.add_argument("--dpi", type=int, default=300,
                         help="Print resolution to generate at (300 is standard/sharp).")
    parser.add_argument("--out", type=str, default="checkerboard.png")
    args = parser.parse_args()

    px_per_mm = args.dpi / 25.4
    square_px = int(round(args.square_mm * px_per_mm))
    border_px = square_px  # one extra square's worth of white border on each side

    board_w = SQUARES_X * square_px
    board_h = SQUARES_Y * square_px
    footer_h = int(1.5 * square_px)  # room for the printed label text

    img_w = board_w + 2 * border_px
    img_h = board_h + 2 * border_px + footer_h

    img = np.full((img_h, img_w), 255, dtype=np.uint8)

    for row in range(SQUARES_Y):
        for col in range(SQUARES_X):
            if (row + col) % 2 == 0:
                y0 = border_px + row * square_px
                x0 = border_px + col * square_px
                img[y0:y0 + square_px, x0:x0 + square_px] = 0

    label = (f"{SQUARES_X-1}x{SQUARES_Y-1} internal corners, {args.square_mm:.0f}mm squares "
             f"-- PRINT AT 100% / ACTUAL SIZE, then verify with a ruler")
    cv2.putText(img, label, (border_px, img_h - int(0.4 * square_px)),
                cv2.FONT_HERSHEY_SIMPLEX, square_px / 220, (128,), 2, cv2.LINE_AA)

    cv2.imwrite(args.out, img)
    print(f"Saved '{args.out}' ({img_w}x{img_h}px at {args.dpi} DPI).")
    print(f"Pattern is {SQUARES_X-1}x{SQUARES_Y-1} internal corners, "
          f"{args.square_mm:.0f}mm squares nominal "
          f"(~{SQUARES_X*args.square_mm:.0f}x{SQUARES_Y*args.square_mm:.0f}mm overall).")
    print("Print at 100% / actual size (landscape), mount on something rigid, "
          "then measure a square with a ruler before calibrating -- if it's off from "
          f"{args.square_mm:.0f}mm, use the real measured value with "
          "stereo_calibrate.py's --square-size-mm.")


if __name__ == "__main__":
    main()
