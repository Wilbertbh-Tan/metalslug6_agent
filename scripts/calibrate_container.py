#!/usr/bin/env python3
"""Calibrate in-game pixel checks inside a running container.

Captures the RetroArch screen, samples pixel values at a grid of positions,
and suggests good in-game check coordinates.  Also saves a screenshot and
a pixel-value heatmap for visual inspection.

Usage (from docker exec):
    docker exec <container> python3 scripts/calibrate_container.py

    # With explicit region:
    docker exec <container> python3 scripts/calibrate_container.py \
        --capture-left 2 --capture-top 40 --capture-width 1278 --capture-height 680

    # Save screenshot + pixel map:
    docker exec <container> python3 scripts/calibrate_container.py --save-screenshot
"""

import argparse
import json
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import cv2
import mss
import numpy as np

from src.env.window_detect import detect_retroarch_window


def capture_gray(left: int, top: int, width: int, height: int) -> np.ndarray:
    """Grab a screenshot and return as grayscale numpy array."""
    with mss.mss() as sct:
        monitor = {"left": left, "top": top, "width": width, "height": height}
        img = sct.grab(monitor)
        frame = np.array(img)[:, :, :3]  # drop alpha
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return gray


def sample_grid(gray: np.ndarray, step_x: int = 40, step_y: int = 40) -> list[dict]:
    """Sample pixel values on a grid across the frame."""
    h, w = gray.shape
    samples = []
    for y in range(0, h, step_y):
        for x in range(0, w, step_x):
            val = int(gray[y, x])
            samples.append({"x": x, "y": y, "value": val})
    return samples


def suggest_in_game_checks(
    samples: list[dict], gray: np.ndarray
) -> list[dict]:
    """Analyze samples and suggest candidate in-game checks.

    Looks for regions that are NOT black (value > 30) and NOT white (< 250),
    which typically indicate HUD elements during gameplay.
    """
    h, w = gray.shape
    candidates = []

    # Focus on top portion of screen (HUD area, y < 15% of height).
    hud_height = int(h * 0.15)
    hud_samples = [s for s in samples if s["y"] < hud_height and 30 < s["value"] < 250]

    # Group by approximate brightness bands.
    for s in hud_samples:
        val = s["value"]
        # Give a range of +/- 40 around the observed value.
        min_v = max(0, val - 40)
        max_v = min(255, val + 40)
        candidates.append({
            "x": s["x"],
            "y": s["y"],
            "observed": val,
            "suggested_min": min_v,
            "suggested_max": max_v,
        })

    # Also sample the middle area for continue-screen detection.
    mid_y = h // 2
    mid_x = w // 2
    mid_samples = [
        s for s in samples
        if abs(s["y"] - mid_y) < h * 0.15
        and abs(s["x"] - mid_x) < w * 0.2
        and s["value"] > 30
    ]
    for s in mid_samples:
        val = s["value"]
        candidates.append({
            "x": s["x"],
            "y": s["y"],
            "observed": val,
            "suggested_min": max(0, val - 40),
            "suggested_max": min(255, val + 40),
            "note": "mid-screen (possible continue area)",
        })

    return candidates


def main():
    parser = argparse.ArgumentParser(description="Calibrate pixel checks inside container")
    parser.add_argument("--capture-left", type=int, default=None)
    parser.add_argument("--capture-top", type=int, default=None)
    parser.add_argument("--capture-width", type=int, default=None)
    parser.add_argument("--capture-height", type=int, default=None)
    parser.add_argument("--no-auto-detect", action="store_true", help="Skip xdotool auto-detection")
    parser.add_argument("--grid-step", type=int, default=40, help="Pixel sampling grid step size")
    parser.add_argument(
        "--num-captures",
        type=int,
        default=3,
        help="Number of captures to average (reduces noise)",
    )
    parser.add_argument("--capture-delay", type=float, default=0.5, help="Delay between captures")
    parser.add_argument("--save-screenshot", action="store_true", help="Save screenshot to outputs/calibration/")
    parser.add_argument(
        "--output-json",
        type=str,
        default="outputs/calibration/container_checks.json",
        help="Path to save suggested checks JSON",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Show top N candidate check positions",
    )
    args = parser.parse_args()

    # Resolve capture region.
    left, top, width, height = 0, 0, 1280, 720  # Xvfb default fallback
    source = "default"

    if not args.no_auto_detect:
        detected = detect_retroarch_window()
        if detected:
            left = detected["left"]
            top = detected["top"]
            width = detected["width"]
            height = detected["height"]
            source = "xdotool_auto"
            print(f"Auto-detected RetroArch window: left={left} top={top} width={width} height={height}")
        else:
            print("Could not auto-detect RetroArch window; using defaults.")

    if args.capture_left is not None:
        left = args.capture_left
        source = "cli"
    if args.capture_top is not None:
        top = args.capture_top
        source = "cli"
    if args.capture_width is not None:
        width = args.capture_width
        source = "cli"
    if args.capture_height is not None:
        height = args.capture_height
        source = "cli"

    print(f"Capture region: left={left} top={top} width={width} height={height} (source={source})")

    # Capture multiple frames and average to reduce noise.
    print(f"Capturing {args.num_captures} frame(s)...")
    frames = []
    for i in range(args.num_captures):
        gray = capture_gray(left, top, width, height)
        frames.append(gray.astype(np.float32))
        if i < args.num_captures - 1:
            time.sleep(args.capture_delay)

    avg_gray = np.mean(frames, axis=0).astype(np.uint8)
    h, w = avg_gray.shape
    print(f"Captured frame: {w}x{h}")

    # Sample grid.
    samples = sample_grid(avg_gray, step_x=args.grid_step, step_y=args.grid_step)

    # Print pixel value overview for HUD area (top 15%).
    hud_cutoff = int(h * 0.15)
    print(f"\n--- HUD area (y < {hud_cutoff}) pixel samples ---")
    print(f"{'x':>5} {'y':>5} {'value':>6}")
    hud_samples = [s for s in samples if s["y"] < hud_cutoff]
    for s in sorted(hud_samples, key=lambda s: (-s["value"], s["x"])):
        marker = " <-- bright" if s["value"] > 100 else ""
        print(f"{s['x']:>5} {s['y']:>5} {s['value']:>6}{marker}")

    # Suggest checks.
    candidates = suggest_in_game_checks(samples, avg_gray)

    # Rank: prefer HUD-area, non-extreme brightness, away from edges.
    def rank(c):
        val = c["observed"]
        in_hud = 1 if c["y"] < hud_cutoff else 0
        brightness_score = min(val, 255 - val)  # prefer mid-range
        edge_penalty = min(c["x"], w - c["x"], c["y"], h - c["y"])
        return (in_hud, brightness_score, edge_penalty)

    candidates.sort(key=rank, reverse=True)
    top_candidates = candidates[: args.top_n]

    print(f"\n--- Top {args.top_n} suggested in-game check positions ---")
    print(f"{'x':>5} {'y':>5} {'observed':>9} {'min':>5} {'max':>5}  note")
    for c in top_candidates:
        note = c.get("note", "HUD area")
        print(
            f"{c['x']:>5} {c['y']:>5} {c['observed']:>9} "
            f"{c['suggested_min']:>5} {c['suggested_max']:>5}  {note}"
        )

    # Format as ready-to-use check tuples.
    if top_candidates:
        print("\n--- Ready-to-use check values ---")
        print("For smoke_env.py --in-game-checks flag:")
        parts = []
        # Pick top 2 HUD candidates for in-game checks.
        hud_cands = [c for c in top_candidates if c.get("note", "") != "mid-screen (possible continue area)"][:2]
        mid_cands = [c for c in top_candidates if "mid-screen" in c.get("note", "")][:1]

        if hud_cands:
            in_game_str = ";".join(
                f"{c['x']},{c['y']},{c['suggested_min']},{c['suggested_max']}"
                for c in hud_cands
            )
            print(f'  --in-game-checks "{in_game_str}"')
        if mid_cands:
            cont_str = ";".join(
                f"{c['x']},{c['y']},{c['suggested_min']},{c['suggested_max']}"
                for c in mid_cands
            )
            print(f'  --continue-checks "{cont_str}"')

    # Save outputs.
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)

    output = {
        "capture_region": {"left": left, "top": top, "width": width, "height": height, "source": source},
        "frame_size": {"width": w, "height": h},
        "candidates": top_candidates,
        "all_samples": samples,
    }
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved calibration data to {args.output_json}")

    if args.save_screenshot:
        screenshot_dir = os.path.dirname(args.output_json) or "outputs/calibration"
        os.makedirs(screenshot_dir, exist_ok=True)
        screenshot_path = os.path.join(screenshot_dir, "container_screenshot.png")
        cv2.imwrite(screenshot_path, avg_gray)
        print(f"Saved screenshot to {screenshot_path}")

        # Also save a color version.
        with mss.mss() as sct:
            monitor = {"left": left, "top": top, "width": width, "height": height}
            img = sct.grab(monitor)
            color = np.array(img)[:, :, :3]
            color_path = os.path.join(screenshot_dir, "container_screenshot_color.png")
            cv2.imwrite(color_path, color)
            print(f"Saved color screenshot to {color_path}")

    print("\nDone. Use the suggested checks with smoke_env.py or train_ppo.py.")


if __name__ == "__main__":
    main()
