import json
import os
from typing import Any


def _parse_int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def load_region_values(
    calibration_json_path: str = "outputs/calibration/region.json",
    use_calibration: bool = True,
    default_left: int = 1116,
    default_top: int = 345,
    default_width: int = 640,
    default_height: int = 480,
) -> dict[str, Any]:
    """
    Resolve capture region with priority:
    1) defaults
    2) calibration JSON (if enabled and present)
    3) CAPTURE_* env vars (explicit override)
    """
    resolved = {
        "left": default_left,
        "top": default_top,
        "width": default_width,
        "height": default_height,
        "source": "default",
        "calibration_path": calibration_json_path,
    }

    if use_calibration and os.path.exists(calibration_json_path):
        try:
            with open(calibration_json_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            region = payload.get("region", {})
            resolved["left"] = int(region.get("left", resolved["left"]))
            resolved["top"] = int(region.get("top", resolved["top"]))
            resolved["width"] = int(region.get("width", resolved["width"]))
            resolved["height"] = int(region.get("height", resolved["height"]))
            resolved["source"] = "calibration_json"
        except Exception:
            # Keep defaults if calibration is unreadable.
            pass

    # Env vars always win when provided.
    resolved["left"] = _parse_int_env("CAPTURE_LEFT", int(resolved["left"]))
    resolved["top"] = _parse_int_env("CAPTURE_TOP", int(resolved["top"]))
    resolved["width"] = _parse_int_env("CAPTURE_WIDTH", int(resolved["width"]))
    resolved["height"] = _parse_int_env("CAPTURE_HEIGHT", int(resolved["height"]))

    if any(os.environ.get(k) for k in ("CAPTURE_LEFT", "CAPTURE_TOP", "CAPTURE_WIDTH", "CAPTURE_HEIGHT")):
        resolved["source"] = "env_override"

    if resolved["width"] <= 0 or resolved["height"] <= 0:
        raise ValueError(
            f"Invalid capture size: width={resolved['width']} height={resolved['height']}"
        )

    return resolved


def load_in_game_checks(
    calibration_json_path: str = "outputs/calibration/region.json",
    use_calibration: bool = True,
    default_checks: list[tuple[int, int, int, int]] | None = None,
) -> list[tuple[int, int, int, int]]:
    checks = list(default_checks or [])
    if not use_calibration or not os.path.exists(calibration_json_path):
        return checks

    try:
        with open(calibration_json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        loaded = payload.get("in_game_checks")
        if not isinstance(loaded, list):
            return checks

        parsed = []
        for item in loaded:
            if not isinstance(item, (list, tuple)) or len(item) != 4:
                continue
            x, y, min_v, max_v = [int(v) for v in item]
            parsed.append((x, y, min_v, max_v))
        if parsed:
            return parsed
    except Exception:
        pass

    return checks
