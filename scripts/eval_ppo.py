import argparse
import json
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.agents.random_agent import RandomAgent
from src.env.mslug_env import CaptureRegion, MetalSlugEnv
from src.env.region_config import load_in_game_checks, load_region_values
from src.env.window_detect import detect_retroarch_window


def make_env(args, verbose=0, fast_forward_mode="off"):
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
        fast_forward_key=args.fast_forward_key,
        fast_forward_mode=fast_forward_mode,
        fast_forward_state_file=args.fast_forward_state_file,
        verbose=verbose,
    )


def main():
    parser = argparse.ArgumentParser(description="Inference-only evaluation for a trained PPO model.")
    parser.add_argument("--model", type=str, default=None, help="Path to .zip model checkpoint")
    parser.add_argument("--random", action="store_true", help="Use random agent instead of a trained model")
    parser.add_argument("--manual", action="store_true",
                        help="Manual mode: no agent actions, just observe rewards from human play")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to evaluate")
    parser.add_argument("--max-steps", type=int, default=5000, help="Max steps per episode")
    parser.add_argument("--sleep", type=float, default=0.02, help="Sleep between steps (for watchable playback)")
    parser.add_argument("--fast-forward-key", type=str, default=None, metavar="KEY",
                        help="Optional key to press once before eval (e.g. space)")
    parser.add_argument(
        "--fast-forward-mode",
        type=str,
        choices=["off", "set_once", "set_once_persist", "on_reset_once", "on_reset_every"],
        default=None,
        help="Fast-forward behavior (default auto: set_once when key is provided)",
    )
    parser.add_argument(
        "--fast-forward-state-file",
        type=str,
        default="outputs/calibration/fast_forward_state.json",
        help="State file used by set_once_persist mode",
    )
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic policy actions (default is deterministic)")
    parser.add_argument("--verbose-env", action="store_true",
                        help="Enable verbose env logging")
    parser.add_argument("--verbose-level", type=int, default=int(os.environ.get("VERBOSE_LEVEL", "1")),
                        help="Verbosity level (0-3): 0 quiet, 1 basic events, 2 reward detail, 3 max diagnostics")
    parser.add_argument(
        "--calibration-json",
        type=str,
        default="outputs/calibration/region.json",
        help="Calibration JSON path produced by scripts/calibrate_region.py",
    )
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Ignore calibration JSON and use env/default capture values only",
    )
    parser.add_argument("--no-auto-detect", action="store_true",
                        help="Disable xdotool window auto-detection")
    parser.add_argument("--disable-in-game-checks", action="store_true",
                        help="Bypass in-game pixel gate")
    parser.add_argument("--capture-left", type=int, default=None)
    parser.add_argument("--capture-top", type=int, default=None)
    parser.add_argument("--capture-width", type=int, default=None)
    parser.add_argument("--capture-height", type=int, default=None)
    parser.add_argument("--high-score-json", type=str, default=None, metavar="PATH",
                        help="Path to a JSON file where the highest score is logged after each episode")
    args = parser.parse_args()

    in_container = os.environ.get("DISPLAY") and os.environ.get("MSLUG_HEADLESS")
    if not in_container:
        print("Click on RetroArch and load your save state (F4).")
        print("Evaluation starts in 5 seconds...")
        time.sleep(5)
    else:
        print("Headless/container mode: using DISPLAY=%s" % os.environ.get("DISPLAY", ""))

    fast_forward_mode = args.fast_forward_mode
    if fast_forward_mode is None:
        fast_forward_mode = "set_once" if args.fast_forward_key else "off"
    print("Fast-forward: key=%s mode=%s" % (args.fast_forward_key, fast_forward_mode))

    if not args.random and not args.manual and not args.model:
        parser.error("--model is required unless --random or --manual is specified")

    # Env internal verbosity: only raise when --verbose-env is set.
    # The eval script handles its own per-step output via --verbose-level.
    env_verbose = max(args.verbose_level, 2) if args.verbose_env else 0
    env = make_env(args, verbose=env_verbose, fast_forward_mode=fast_forward_mode)

    if args.random:
        agent = RandomAgent(env.action_space)
        mode_label = "random"
    elif args.manual:
        agent = None
        mode_label = "manual (noop)"
    else:
        from src.agents.ppo_agent import PPOAgent
        agent = PPOAgent(args.model, deterministic=not args.stochastic)
        mode_label = f"PPO deterministic={agent.deterministic}"

    MOVEMENT_NAMES = ["noop", "left", "right", "up", "down"]
    ATTACK_NAMES = ["noop", "shoot", "grenade"]
    MODIFIER_NAMES = ["noop", "jump", "slug_atk"]

    def format_action(a):
        """Format a MultiDiscrete action as readable string."""
        import numpy as np
        a = np.asarray(a).flatten()
        if len(a) >= 3:
            parts = [n for n in [MOVEMENT_NAMES[a[0]], ATTACK_NAMES[a[1]], MODIFIER_NAMES[a[2]]] if n != "noop"]
            return "+".join(parts) or "noop"
        return str(a)

    episode_rewards = []
    episode_lengths = []
    episode_scores = []
    high_score = 0
    high_score_ep = 0
    run_indefinitely = args.episodes <= 0
    verbose = args.verbose_level

    if run_indefinitely:
        print(f"Running indefinitely (Ctrl+C to stop) | mode={mode_label}")
    else:
        print(f"Running {args.episodes} episode(s) | mode={mode_label}")

    ep = 0
    try:
        while run_indefinitely or ep < args.episodes:
            obs, _ = env.reset()
            ep_reward = 0.0
            ep_score_reward = 0.0
            ep_progress_reward = 0.0
            ep_time_penalty = 0.0
            ep_len = 0
            ep_score = 0
            ep_start = time.monotonic()
            done = False

            while not done and ep_len < args.max_steps:
                action = agent.predict(obs) if agent else np.array([0, 0, 0])  # noop in manual mode
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += float(reward)
                ep_len += 1
                done = bool(terminated or truncated)

                ep_score = info.get("score", ep_score)
                ep_score_reward += info.get("score_reward", 0.0)
                ep_progress_reward += info.get("progress_reward", 0.0)
                ep_time_penalty += info.get("time_penalty", 0.0)

                if verbose >= 2:
                    t = time.monotonic() - ep_start
                    act_name = format_action(action)
                    print(
                        f"  [{t:6.1f}s] ep={ep + 1} step={ep_len:>4} "
                        f"act={act_name:<20s} "
                        f"rew={reward:+.4f} total={ep_reward:+.4f} "
                        f"score={ep_score:>8} "
                        f"Rscore={info.get('score_reward', 0):+.4f} "
                        f"Rprog={info.get('progress_reward', 0):+.4f} "
                        f"Rtime={info.get('time_penalty', 0):+.5f}"
                    )

                time.sleep(args.sleep)

            ep += 1
            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_len)
            episode_scores.append(ep_score)
            if ep_score > high_score:
                high_score = ep_score
                high_score_ep = ep

            reason = info.get("terminate_reason", "max_steps") if done else "max_steps"
            summary = f"Episode {ep}: reward={ep_reward:+.4f}, length={ep_len}, score={ep_score}, end={reason}, high_score={high_score} (ep {high_score_ep})"
            if verbose >= 1:
                summary += (
                    f"\n  breakdown: score_reward={ep_score_reward:+.4f} "
                    f"progress_reward={ep_progress_reward:+.4f} "
                    f"time_penalty={ep_time_penalty:+.4f}"
                )
            print(summary)

            if args.high_score_json:
                hs_dir = os.path.dirname(args.high_score_json)
                if hs_dir:
                    os.makedirs(hs_dir, exist_ok=True)
                hs_data = {
                    "high_score": high_score,
                    "high_score_episode": high_score_ep,
                    "total_episodes": ep,
                    "last_score": ep_score,
                    "last_reward": round(ep_reward, 4),
                }
                with open(args.high_score_json, "w", encoding="utf-8") as f:
                    json.dump(hs_data, f, indent=2)
    except KeyboardInterrupt:
        print()

    if episode_rewards:
        avg_rew = sum(episode_rewards) / len(episode_rewards)
        avg_len = sum(episode_lengths) / len(episode_lengths)
        avg_score = sum(episode_scores) / len(episode_scores)
        print(f"\nEvaluation complete ({ep} episodes)")
        print(f"Average reward: {avg_rew:+.4f}")
        print(f"Average score:  {avg_score:.0f}")
        print(f"Average length: {avg_len:.1f}")
        print(f"High score:     {high_score} (episode {high_score_ep})")

    env.close()


if __name__ == "__main__":
    main()
