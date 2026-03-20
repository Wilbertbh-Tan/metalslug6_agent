"""Auto-detect RetroArch window geometry inside a container (Xvfb + xdotool)."""

import os
import shutil
import subprocess

import mss
import numpy as np


def _get_screen_size(env: dict) -> tuple[int, int]:
    """Get the Xvfb screen dimensions via xdpyinfo or fallback to 1280x720."""
    try:
        out = subprocess.check_output(
            ["xdpyinfo"],
            env=env,
            timeout=5,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        for line in out.splitlines():
            if "dimensions:" in line:
                # e.g. "  dimensions:    1280x720 pixels (338x190 millimeters)"
                dim = line.split("dimensions:")[1].strip().split()[0]
                w, h = dim.split("x")
                return int(w), int(h)
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        FileNotFoundError,
    ):
        pass
    return 1280, 720


def _crop_black_borders(
    left: int,
    top: int,
    width: int,
    height: int,
    threshold: float = 10.0,
) -> tuple[int, int, int, int]:
    """Grab the window region and trim black border rows/columns.

    Returns adjusted (left, top, width, height).  Falls back to the
    original values if the grab fails or the frame is entirely dark.
    """
    try:
        sct = mss.mss()
        region = {"left": left, "top": top, "width": width, "height": height}
        frame = np.array(sct.grab(region))[:, :, :3]
        sct.close()
    except Exception:
        return left, top, width, height

    gray = np.mean(frame, axis=2)
    col_means = np.mean(gray, axis=0)
    row_means = np.mean(gray, axis=1)

    non_black_cols = np.where(col_means > threshold)[0]
    non_black_rows = np.where(row_means > threshold)[0]

    if len(non_black_cols) == 0 or len(non_black_rows) == 0:
        return left, top, width, height

    x1, x2 = int(non_black_cols[0]), int(non_black_cols[-1])
    y1, y2 = int(non_black_rows[0]), int(non_black_rows[-1])

    return left + x1, top + y1, x2 - x1 + 1, y2 - y1 + 1


def detect_retroarch_window() -> dict | None:
    """Find the RetroArch X11 window and return its geometry.

    Returns dict with left, top, width, height, source or None if not found.
    Clamps to screen bounds so mss.grab() never exceeds the framebuffer.
    """
    if not shutil.which("xdotool"):
        return None

    display = os.environ.get("DISPLAY", ":99")
    env = {**os.environ, "DISPLAY": display}

    # Try by name first (most reliable inside containers).
    win_id = None
    for search in [["xdotool", "search", "--name", "RetroArch"]]:
        try:
            out = subprocess.check_output(search, env=env, timeout=5, text=True).strip()
            if out:
                win_id = out.splitlines()[0]
                break
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            continue

    if not win_id:
        return None

    # Get geometry.
    try:
        out = subprocess.check_output(
            ["xdotool", "getwindowgeometry", "--shell", win_id],
            env=env,
            timeout=5,
            text=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None

    geo = {}
    for line in out.strip().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            geo[k.strip()] = v.strip()

    try:
        left = max(0, int(geo.get("X", 0)))
        top = max(0, int(geo.get("Y", 0)))
        width = int(geo.get("WIDTH", 0))
        height = int(geo.get("HEIGHT", 0))
    except (ValueError, TypeError):
        return None

    if width <= 0 or height <= 0:
        return None

    # Clamp to screen bounds so XGetImage/mss never reads past the framebuffer.
    screen_w, screen_h = _get_screen_size(env)
    if left + width > screen_w:
        width = screen_w - left
    if top + height > screen_h:
        height = screen_h - top

    if width <= 0 or height <= 0:
        return None

    # Crop black borders (letterbox/pillarbox) from the captured window.
    left, top, width, height = _crop_black_borders(left, top, width, height)

    if width <= 0 or height <= 0:
        return None

    return {
        "left": left,
        "top": top,
        "width": width,
        "height": height,
        "source": "xdotool_auto",
    }
