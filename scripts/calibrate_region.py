import argparse
import json
import os
import time
from datetime import datetime

import cv2
import mss
import numpy as np
import pyautogui


COMMON_SIZES = [
    (640, 480),
    (640, 448),
    (640, 440),
    (640, 432),
]

HUD_ANCHORS = [
    ("arms_a_top", "top of the 'A' in ARMS"),
    ("gun_silver_border_top_left", "top-left of the silver gun border icon"),
    ("oneup_u_top_left", "top-left of the 'U' in 1UP"),
]


def capture_point_countdown(label: str, countdown_s: int):
    print(f"\nMove mouse to {label}. Capturing in {countdown_s} seconds...")
    for i in range(countdown_s, 0, -1):
        print(f"  {i}...")
        time.sleep(1)
    pos = pyautogui.position()
    print(f"Captured {label}: ({pos.x}, {pos.y})")
    return pos.x, pos.y


def capture_point_enter(label: str):
    input(f"\nMove mouse to {label}, then press Enter to capture...")
    pos = pyautogui.position()
    print(f"Captured {label}: ({pos.x}, {pos.y})")
    return pos.x, pos.y


def round_to_multiple(value: int, multiple: int) -> int:
    return max(multiple, int(round(value / multiple) * multiple))


def choose_common_size(width: int, height: int, tolerance: int):
    best = None
    best_dist = None
    for w, h in COMMON_SIZES:
        dist = abs(w - width) + abs(h - height)
        if best is None or dist < best_dist:
            best = (w, h)
            best_dist = dist
    if best is None or best_dist is None or best_dist > tolerance:
        return None
    return best


def adjust_region(left: int, top: int, width: int, height: int, snap_multiple: int, common_tol: int):
    # Keep the center stable when snapping dimensions, so errors are distributed.
    cx = left + (width / 2.0)
    cy = top + (height / 2.0)

    snapped_w = round_to_multiple(width, snap_multiple)
    snapped_h = round_to_multiple(height, snap_multiple)

    common = choose_common_size(snapped_w, snapped_h, common_tol)
    if common is not None:
        snapped_w, snapped_h = common

    snapped_left = int(round(cx - (snapped_w / 2.0)))
    snapped_top = int(round(cy - (snapped_h / 2.0)))

    if snapped_left < 0:
        snapped_left = 0
    if snapped_top < 0:
        snapped_top = 0

    return snapped_left, snapped_top, snapped_w, snapped_h


def _capture_point(label: str, method: str, countdown_s: int):
    if method == "countdown":
        return capture_point_countdown(label, countdown_s)
    return capture_point_enter(label)


def _sample_gray_at_abs(region: dict, abs_x: int, abs_y: int) -> int | None:
    rel_x = abs_x - int(region["left"])
    rel_y = abs_y - int(region["top"])
    if rel_x < 0 or rel_y < 0 or rel_x >= int(region["width"]) or rel_y >= int(region["height"]):
        return None
    with mss.mss() as sct:
        frame = np.array(sct.grab(region))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    return int(gray[rel_y, rel_x])


def main():
    parser = argparse.ArgumentParser(description="Calibrate game capture region from mouse points.")
    parser.add_argument(
        "--method",
        choices=["countdown", "enter"],
        default="countdown",
        help="How to capture each point (default: countdown)",
    )
    parser.add_argument(
        "--countdown",
        type=int,
        default=5,
        help="Seconds before each capture in countdown mode (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/calibration/region.json",
        help="Output JSON path (default: outputs/calibration/region.json)",
    )
    parser.add_argument(
        "--no-auto-adjust",
        action="store_true",
        help="Disable dimension snapping/normalization",
    )
    parser.add_argument(
        "--snap-multiple",
        type=int,
        default=8,
        help="Round width/height to nearest multiple of this value (default: 8)",
    )
    parser.add_argument(
        "--common-size-tolerance",
        type=int,
        default=24,
        help="Max distance for snapping to common sizes like 640x440 (default: 24)",
    )
    parser.add_argument(
        "--check-tolerance",
        type=int,
        default=25,
        help="Tolerance around sampled HUD grayscale for generated in-game checks (default: 25)",
    )
    parser.add_argument(
        "--skip-icon-checks",
        action="store_true",
        help="Skip HUD icon sampling and do region-only calibration",
    )
    args = parser.parse_args()

    if args.method == "countdown":
        p1x, p1y = capture_point_countdown("TOP-LEFT corner of the game area", args.countdown)
        p2x, p2y = capture_point_countdown("BOTTOM-RIGHT corner of the game area", args.countdown)
    else:
        p1x, p1y = capture_point_enter("TOP-LEFT corner of the game area")
        p2x, p2y = capture_point_enter("BOTTOM-RIGHT corner of the game area")

    left = min(p1x, p2x)
    top = min(p1y, p2y)
    right = max(p1x, p2x)
    bottom = max(p1y, p2y)
    width = right - left
    height = bottom - top

    if width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid region computed: left={left}, top={top}, width={width}, height={height}. "
            "Please run calibration again."
        )

    raw_region = {"left": left, "top": top, "width": width, "height": height}

    if args.no_auto_adjust:
        adj_left, adj_top, adj_width, adj_height = left, top, width, height
        adjusted = False
        adjust_reason = "disabled"
    else:
        adj_left, adj_top, adj_width, adj_height = adjust_region(
            left,
            top,
            width,
            height,
            snap_multiple=args.snap_multiple,
            common_tol=args.common_size_tolerance,
        )
        adjusted = (adj_left != left) or (adj_top != top) or (adj_width != width) or (adj_height != height)
        adjust_reason = "auto_snap"

    adjusted_region = {
        "left": adj_left,
        "top": adj_top,
        "width": adj_width,
        "height": adj_height,
    }

    anchors = []
    generated_in_game_checks = []
    if not args.skip_icon_checks:
        print("\nHUD icon calibration (for in-game checks):")
        print("Keep the game on a normal in-game screen while capturing these points.")
        for anchor_key, label in HUD_ANCHORS:
            ax, ay = _capture_point(label, args.method, args.countdown)
            sampled = _sample_gray_at_abs(adjusted_region, ax, ay)
            if sampled is None:
                print(
                    f"  WARN: {anchor_key} at ({ax},{ay}) is outside adjusted region; "
                    "skipping this anchor."
                )
                anchors.append(
                    {
                        "name": anchor_key,
                        "label": label,
                        "screen": {"x": ax, "y": ay},
                        "relative": None,
                        "gray_value": None,
                        "included_in_checks": False,
                    }
                )
                continue

            rel_x = ax - adj_left
            rel_y = ay - adj_top
            min_v = max(0, sampled - args.check_tolerance)
            max_v = min(255, sampled + args.check_tolerance)
            generated_in_game_checks.append((rel_x, rel_y, min_v, max_v))
            anchors.append(
                {
                    "name": anchor_key,
                    "label": label,
                    "screen": {"x": ax, "y": ay},
                    "relative": {"x": rel_x, "y": rel_y},
                    "gray_value": sampled,
                    "included_in_checks": True,
                    "check": [rel_x, rel_y, min_v, max_v],
                }
            )

    payload = {
        "captured_at": datetime.utcnow().isoformat() + "Z",
        "method": args.method,
        "top_left": {"x": adj_left, "y": adj_top},
        "bottom_right": {"x": adj_left + adj_width, "y": adj_top + adj_height},
        "region": {"left": adj_left, "top": adj_top, "width": adj_width, "height": adj_height},
        "raw_region": raw_region,
        "adjustment": {
            "applied": adjusted,
            "reason": adjust_reason,
            "snap_multiple": args.snap_multiple,
            "common_size_tolerance": args.common_size_tolerance,
            "common_sizes": COMMON_SIZES,
        },
        "anchors": anchors,
        "in_game_checks": generated_in_game_checks,
        "in_game_checks_meta": {
            "source": "calibrated_hud_anchors",
            "tolerance": args.check_tolerance,
            "count": len(generated_in_game_checks),
        },
    }

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\nCalibration complete.")
    print(f"Raw region: left={left}, top={top}, width={width}, height={height}")
    if adjusted:
        print(
            f"Adjusted region: left={adj_left}, top={adj_top}, width={adj_width}, height={adj_height} "
            "(auto-snapped)"
        )
    else:
        print(f"Adjusted region: left={adj_left}, top={adj_top}, width={adj_width}, height={adj_height}")
    print(f"Saved: {args.output}")
    print("\nUse in code:")
    print(f"  CaptureRegion(left={adj_left}, top={adj_top}, width={adj_width}, height={adj_height})")
    print("\nOr via env (for training/eval):")
    print(f"  export CAPTURE_LEFT={adj_left}")
    print(f"  export CAPTURE_TOP={adj_top}")
    print(f"  export CAPTURE_WIDTH={adj_width}")
    print(f"  export CAPTURE_HEIGHT={adj_height}")
    if generated_in_game_checks:
        print("\nGenerated in-game checks from HUD anchors:")
        for check in generated_in_game_checks:
            print(f"  {check}")


if __name__ == "__main__":
    main()
