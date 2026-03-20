#!/usr/bin/env python3
"""Create a RetroArch save state at the very start of gameplay.

Sends a boot key sequence via xdotool (targeted at the RetroArch window),
then polls RAM until game_state==0x00 (in-game), score==0, and lives==3.
Saves state (F2) at that exact frame.
"""
import os
import subprocess
import sys
import time

os.environ.setdefault("DISPLAY", ":99")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.env.mslug_env import CaptureRegion, MetalSlugEnv  # noqa: E402

SCORE_ADDR = os.environ.get("SCORE_ADDR", "003869BC")
# Boot detection uses 003868D1 (static 'starting lives config' byte, always 0x03).
# The real gameplay lives counter at 003D3B07 may not be populated during BIOS boot.
LIVES_ADDR = os.environ.get("LIVES_ADDR", "003868D1")
GAME_STATE_ADDR = os.environ.get("GAME_STATE_ADDR", "003868D0")
CAPTURE_LEFT = int(os.environ.get("CAPTURE_LEFT", "0"))
CAPTURE_TOP = int(os.environ.get("CAPTURE_TOP", "0"))
CAPTURE_WIDTH = int(os.environ.get("CAPTURE_WIDTH", "640"))
CAPTURE_HEIGHT = int(os.environ.get("CAPTURE_HEIGHT", "440"))


def _parse_key_sequence(raw: str):
    """Parse "key:delay:key:delay:..." into [(key, delay_after), ...]."""
    parts = [p.strip() for p in raw.split(":") if p.strip()]
    if not parts:
        return []
    seq = []
    i = 0
    while i < len(parts):
        key = parts[i]
        delay_after = 0.0
        i += 1
        if i < len(parts):
            try:
                delay_after = float(parts[i])
                i += 1
            except ValueError:
                delay_after = 0.0
        seq.append((key, delay_after))
    return seq


def _find_retroarch_window():
    """Find RetroArch window ID via xdotool."""
    try:
        out = subprocess.check_output(
            ["xdotool", "search", "--name", "RetroArch"],
            stderr=subprocess.DEVNULL, timeout=5,
        )
        for line in out.decode().strip().splitlines():
            wid = line.strip()
            if wid:
                return wid
    except Exception:
        pass
    return None


def _send_key(wid, key):
    """Send a key to the RetroArch window via xdotool."""
    cmd = ["xdotool", "key", "--window", wid, key] if wid else ["xdotool", "key", key]
    subprocess.run(cmd, stderr=subprocess.DEVNULL, timeout=5)


def is_gameplay_start(env):
    """RAM-based check: in-game (game_state==0x00), score==0, and lives>=2.

    The score==0 check is critical to distinguish actual gameplay start from
    the title/menu screen where game_state and lives may hold stale values
    from a previous game.
    """
    gs = env._read_ram(GAME_STATE_ADDR, 1)
    if not gs or len(gs) != 1 or gs[0] != 0x00:
        return False
    score = env._read_ram(SCORE_ADDR, 4)
    if not score or len(score) != 4 or score != [0, 0, 0, 0]:
        return False
    lives = env._read_ram(LIVES_ADDR, 1)
    if not lives or len(lives) != 1 or lives[0] < 0x02:
        return False
    return True


def main():
    wid = _find_retroarch_window()
    print(f"[savestate] RetroArch window: {wid}")

    region = CaptureRegion(
        left=CAPTURE_LEFT, top=CAPTURE_TOP,
        width=CAPTURE_WIDTH, height=CAPTURE_HEIGHT,
    )
    env = MetalSlugEnv(region=region, verbose=0)

    try:
        # Focus the window first.
        if wid:
            subprocess.run(
                ["xdotool", "windowfocus", "--sync", wid],
                stderr=subprocess.DEVNULL, timeout=5,
            )
            time.sleep(0.5)

        # Keep compatibility with documented BOOT_KEYS from CONTAINER.md.
        sequence_raw = os.environ.get("BOOT_KEYS", "Shift_R:2:Return:4:x:4:x:4")
        key_sequence = _parse_key_sequence(sequence_raw)
        print(f"[savestate] Boot sequence: {sequence_raw}")
        for key, delay_after in key_sequence:
            print(f"[savestate] Sending key: {key}")
            _send_key(wid, key)
            if delay_after > 0:
                time.sleep(delay_after)

        # Poll RAM until game_state==0x00, score==0, lives==3.
        print("[savestate] Waiting for gameplay start (RAM: game_state=0, score=0, lives>=2)...")
        wait_timeout = float(os.environ.get("SAVESTATE_WAIT_TIMEOUT", "60"))
        deadline = time.monotonic() + wait_timeout
        saved = False
        attempt = 0
        while time.monotonic() < deadline:
            time.sleep(0.15)
            ready = is_gameplay_start(env)
            attempt += 1
            if attempt % 20 == 0:
                gs = env._read_ram(GAME_STATE_ADDR, 1)
                sc = env._read_ram(SCORE_ADDR, 4)
                lv = env._read_ram(LIVES_ADDR, 1)
                print(f"[savestate]   ...{attempt} polls, game_state={gs}, score={sc}, lives={lv}")
            if ready:
                # Use SAVE_STATE network command (more reliable than F2 key).
                env._send_retroarch_cmd("SAVE_STATE")
                time.sleep(1.0)
                # Verify the save state file was actually created.
                import glob
                state_files = glob.glob("/games/Flycast/*.state") + glob.glob("/games/*.state")
                found = [f for f in state_files if os.path.getsize(f) > 0]
                if found:
                    print(f"[savestate] Save state created: {found[0]} ({os.path.getsize(found[0])} bytes)")
                    saved = True
                    break
                else:
                    # Fallback: try F2 key press.
                    print("[savestate] SAVE_STATE command didn't create file, trying F2 key...")
                    _send_key(wid, "F2")
                    time.sleep(1.0)
                    state_files = glob.glob("/games/Flycast/*.state") + glob.glob("/games/*.state")
                    found = [f for f in state_files if os.path.getsize(f) > 0]
                    if found:
                        print(f"[savestate] Save state created via F2: {found[0]} ({os.path.getsize(found[0])} bytes)")
                        saved = True
                        break
                    print("[savestate] F2 also failed, continuing to poll...")

        if not saved:
            raise RuntimeError(
                "Could not create save state (game_state=0, score=0, lives>=2). "
                "Neither SAVE_STATE command nor F2 key produced a valid state file."
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
