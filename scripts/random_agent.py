#!/usr/bin/env python3
"""Random agent for Metal Slug 6.

Runs random actions in the environment, logs rewards and episode summaries.
Useful for validating death detection, reward signals, and env stability.
"""

import argparse
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.agents.random_agent import RandomAgent
from src.env.mslug_env import CaptureRegion, MetalSlugEnv
from src.env.region_config import load_in_game_checks, load_region_values
from src.env.window_detect import detect_retroarch_window


ACTION_NAMES = {
    0: "noop",
    1: "left",
    2: "right",
    3: "up",
    4: "down",
    5: "shoot",
    6: "jump",
    7: "grenade",
    8: "slug_attack",
    9: "weapon_change",
}


def make_env(args) -> MetalSlugEnv:
    """Create the environment with auto-detected or CLI-specified region."""
    region_values = load_region_values(
        calibration_json_path=args.calibration_json,
        use_calibration=not args.no_calibration,
    )
    if not args.no_auto_detect:
        detected = detect_retroarch_window()
        if detected:
            print(f"Auto-detected RetroArch window: {detected}")
            region_values.update(detected)

    if args.capture_left is not None:
        region_values["left"] = args.capture_left
    if args.capture_top is not None:
        region_values["top"] = args.capture_top
    if args.capture_width is not None:
        region_values["width"] = args.capture_width
    if args.capture_height is not None:
        region_values["height"] = args.capture_height

    region = CaptureRegion(
        left=region_values["left"],
        top=region_values["top"],
        width=region_values["width"],
        height=region_values["height"],
    )
    print(
        f"Capture region: {region_values['left']},{region_values['top']} "
        f"{region_values['width']}x{region_values['height']} "
        f"(source={region_values.get('source', 'default')})"
    )

    in_game_checks = load_in_game_checks(
        calibration_json_path=args.calibration_json,
        use_calibration=not args.no_calibration,
        default_checks=[],
    )

    return MetalSlugEnv(
        region=region,
        in_game_checks=in_game_checks,
        enforce_in_game_checks=not args.disable_in_game_checks,
        reset_via_network=True,
        verbose=args.verbose_level,
    )


def main():
    parser = argparse.ArgumentParser(description="Random agent for Metal Slug 6")
    parser.add_argument(
        "--max-steps", type=int, default=0, help="Stop after N steps (0 = unlimited)"
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=0,
        help="Stop after N episodes (0 = unlimited)",
    )
    parser.add_argument(
        "--sleep", type=float, default=0.02, help="Sleep between steps (seconds)"
    )
    parser.add_argument(
        "--verbose-level",
        type=int,
        default=1,
        help="Env verbosity (0=quiet, 1=steps, 2=debug)",
    )
    parser.add_argument(
        "--disable-in-game-checks",
        action="store_true",
        help="Bypass in-game pixel gate",
    )
    parser.add_argument(
        "--no-auto-detect",
        action="store_true",
        help="Disable xdotool window auto-detection",
    )
    parser.add_argument("--capture-left", type=int, default=None)
    parser.add_argument("--capture-top", type=int, default=None)
    parser.add_argument("--capture-width", type=int, default=None)
    parser.add_argument("--capture-height", type=int, default=None)
    parser.add_argument(
        "--calibration-json", type=str, default="outputs/calibration/region.json"
    )
    parser.add_argument("--no-calibration", action="store_true")
    args = parser.parse_args()

    env = make_env(args)
    agent = RandomAgent(env.action_space)
    obs, _ = env.reset()
    print(f"Observation shape: {obs.shape}")
    print()

    episode = 0
    total_steps = 0
    ep_reward = 0.0
    ep_steps = 0
    ep_start = time.monotonic()
    run_indefinitely = args.max_steps <= 0

    try:
        while run_indefinitely or total_steps < args.max_steps:
            action = agent.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_steps += 1
            ep_steps += 1
            ep_reward += reward

            phase = info.get("phase", "unknown")
            score = info.get("score", 0)
            t = time.monotonic() - ep_start

            if args.verbose_level >= 1:
                print(
                    f"[{t:6.1f}s] step={ep_steps:>4} "
                    f"act={ACTION_NAMES[action]:<16s} "
                    f"score={score:>8} rew={reward:+.4f} "
                    f"total={ep_reward:+.4f} phase={phase}"
                )

            if terminated or truncated:
                episode += 1
                ep_elapsed = time.monotonic() - ep_start
                reason = info.get("terminate_reason", "unknown")
                print(f"\n{'=' * 50}")
                print(f"Episode {episode} ended: reason={reason}")
                print(
                    f"  steps={ep_steps}  score={score}  "
                    f"reward={ep_reward:+.4f}  time={ep_elapsed:.1f}s"
                )
                print(f"{'=' * 50}\n")

                if args.max_episodes > 0 and episode >= args.max_episodes:
                    print(f"Reached max_episodes={args.max_episodes}. Done.")
                    break

                obs, _ = env.reset()
                ep_reward = 0.0
                ep_steps = 0
                ep_start = time.monotonic()

            time.sleep(args.sleep)

    except KeyboardInterrupt:
        print(f"\nStopped. episodes={episode} total_steps={total_steps}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
