import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from src.env.mslug_env import CaptureRegion, MetalSlugEnv
from src.env.region_config import load_in_game_checks, load_region_values
from src.env.window_detect import detect_retroarch_window


class ScoreLoggingCallback(BaseCallback):
    """
    Logs score stats during training, including:
    - round average episode score (round = score-log interval)
    - global average episode score
    - global high score and where it happened (episode/step)
    """

    def __init__(self, log_every: int = 500, print_stdout: bool = True):
        super().__init__(verbose=0)
        self.log_every = max(1, int(log_every))
        self.print_stdout = print_stdout
        self.last_score = None
        self.high_score = 0.0
        self.round_index = 0
        self.episode_count = 0
        self.episode_score_sum = 0.0
        self.episode_reward_sum = 0.0
        self.round_episode_scores = []
        self.round_episode_rewards = []
        self.high_score_episode = 0
        self.last_score_raw_hex = None

    def _on_step(self) -> bool:
        infos = self.locals.get("infos") or []
        dones = self.locals.get("dones")
        done_flags = []
        if dones is not None:
            done_flags = list(np.asarray(dones).astype(bool))
        else:
            done_flags = [False] * len(infos)

        for idx, info in enumerate(infos):
            if not info:
                continue
            score = info.get("score")
            score_raw_hex = info.get("score_raw_hex")
            if score_raw_hex is not None:
                self.last_score_raw_hex = str(score_raw_hex)
            if score is None:
                score_value = None
            else:
                score_value = float(score)
                self.last_score = score_value
                if score_value > self.high_score:
                    self.high_score = score_value
                    # Approximate episode index for in-progress episodes.
                    self.high_score_episode = self.episode_count + 1

            is_done = done_flags[idx] if idx < len(done_flags) else False
            if is_done:
                # Terminal info may not include score, so use latest known score.
                final_ep_score = score_value if score_value is not None else float(self.last_score or 0.0)
                ep_reward = info.get("episode_reward")
                ep_steps = info.get("episode_steps", 0)
                self.episode_count += 1
                self.episode_score_sum += final_ep_score
                self.round_episode_scores.append(final_ep_score)
                if ep_reward is not None:
                    self.episode_reward_sum += float(ep_reward)
                    self.round_episode_rewards.append(float(ep_reward))
                if final_ep_score > self.high_score:
                    self.high_score = final_ep_score
                    self.high_score_episode = self.episode_count
                if self.print_stdout:
                    print(
                        "[episode] ep=%s score=%s reward=%s steps=%s reason=%s"
                        % (
                            self.episode_count,
                            ("%.0f" % final_ep_score),
                            ("%.2f" % ep_reward) if ep_reward is not None else "n/a",
                            ep_steps,
                            info.get("terminate_reason", "n/a"),
                        )
                    )

        if self.num_timesteps % self.log_every == 0:
            self.round_index += 1
            round_avg_score = None
            round_avg_reward = None
            if self.round_episode_scores:
                round_avg_score = float(sum(self.round_episode_scores) / len(self.round_episode_scores))
            if self.round_episode_rewards:
                round_avg_reward = float(sum(self.round_episode_rewards) / len(self.round_episode_rewards))
            global_avg_score = None
            global_avg_reward = None
            if self.episode_count > 0:
                global_avg_score = float(self.episode_score_sum / self.episode_count)
                global_avg_reward = float(self.episode_reward_sum / self.episode_count)

            self.logger.record("game/round", float(self.round_index))
            self.logger.record("game/episodes_seen", float(self.episode_count))
            if self.last_score is not None:
                self.logger.record("game/score", self.last_score)
            self.logger.record("game/high_score", self.high_score)
            self.logger.record("game/high_score_episode", float(self.high_score_episode))
            if round_avg_score is not None:
                self.logger.record("game/round_avg_episode_score", round_avg_score)
            if global_avg_score is not None:
                self.logger.record("game/global_avg_episode_score", global_avg_score)
            if round_avg_reward is not None:
                self.logger.record("game/round_avg_episode_reward", round_avg_reward)
            if global_avg_reward is not None:
                self.logger.record("game/global_avg_episode_reward", global_avg_reward)

            if self.print_stdout:
                print(
                    "[monitor] step=%s eps=%s round_avg_score=%s round_avg_reward=%s "
                    "global_avg_score=%s global_avg_reward=%s high_score=%s"
                    % (
                        self.num_timesteps,
                        self.episode_count,
                        ("%.0f" % round_avg_score) if round_avg_score is not None else "n/a",
                        ("%.2f" % round_avg_reward) if round_avg_reward is not None else "n/a",
                        ("%.0f" % global_avg_score) if global_avg_score is not None else "n/a",
                        ("%.2f" % global_avg_reward) if global_avg_reward is not None else "n/a",
                        ("%.0f" % self.high_score),
                    )
                )
            # Start next round window after logging.
            self.round_episode_scores = []
            self.round_episode_rewards = []
        return True


class EpisodeControlCallback(BaseCallback):
    """
    Tracks ended episodes, optionally records periodic evaluation videos,
    and can stop training at a target episode count.
    """

    def __init__(
        self,
        target_episodes: int = 0,
        video_every_episodes: int = 0,
        video_max_steps: int = 800,
        video_fps: int = 20,
        video_dir: str = "outputs/videos",
        eval_env_factory=None,
        verbose: int = 0,
    ):
        super().__init__(verbose=0)
        self.target_episodes = max(0, int(target_episodes))
        self.video_every_episodes = max(0, int(video_every_episodes))
        self.video_max_steps = max(1, int(video_max_steps))
        self.video_fps = max(1, int(video_fps))
        self.video_dir = video_dir
        self.eval_env_factory = eval_env_factory
        self.print_verbose = verbose
        self.episode_count = 0
        self._eval_env = None

    def _log(self, msg: str):
        if self.print_verbose >= 1:
            print(msg)

    def _ensure_eval_env(self):
        if self._eval_env is None and self.eval_env_factory is not None:
            self._eval_env = self.eval_env_factory()
        return self._eval_env

    def _write_video(self, frames_rgb, output_path: str):
        if not frames_rgb:
            return
        h, w, _ = frames_rgb[0].shape
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(self.video_fps),
            (w, h),
        )
        for frame_rgb in frames_rgb:
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
        writer.release()

    def _record_video(self):
        env = self._ensure_eval_env()
        if env is None:
            return
        os.makedirs(self.video_dir, exist_ok=True)

        obs, _ = env.reset()
        frames = []
        frame = env.render()
        if frame is not None:
            frames.append(np.asarray(frame, dtype=np.uint8))

        for _ in range(self.video_max_steps):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            frame = env.render()
            if frame is not None:
                frames.append(np.asarray(frame, dtype=np.uint8))
            if terminated or truncated:
                break

        filename = "episode_%07d_step_%09d.mp4" % (self.episode_count, self.num_timesteps)
        out_path = os.path.join(self.video_dir, filename)
        self._write_video(frames, out_path)
        self._log(f"[video] Saved {out_path} ({len(frames)} frames)")

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        if dones is None:
            return True
        ended_now = int(np.sum(np.asarray(dones, dtype=np.int32)))
        if ended_now <= 0:
            return True

        for _ in range(ended_now):
            self.episode_count += 1
            self.logger.record("game/episodes", float(self.episode_count))
            if self.video_every_episodes > 0 and self.episode_count % self.video_every_episodes == 0:
                self._log(
                    "[video] Episode %s reached at step %s. Recording..."
                    % (self.episode_count, self.num_timesteps)
                )
                self._record_video()

            if self.target_episodes > 0 and self.episode_count >= self.target_episodes:
                self._log(
                    "[episodes] Target reached: %s episodes at step %s. Stopping training."
                    % (self.episode_count, self.num_timesteps)
                )
                return False
        return True

    def _on_training_end(self) -> None:
        if self._eval_env is not None:
            self._eval_env.close()
            self._eval_env = None


def make_env(
    verbose=False,
    fast_forward_key=None,
    fast_forward_mode="off",
    fast_forward_state_file="outputs/calibration/fast_forward_state.json",
    fast_forward_on_reset=False,
    fast_forward_once=True,
    calibration_json_path="outputs/calibration/region.json",
    use_calibration=True,
):
    region_values = load_region_values(
        calibration_json_path=calibration_json_path,
        use_calibration=use_calibration,
    )
    auto_detected = False
    detected = detect_retroarch_window()
    if detected:
        region_values.update(detected)
        auto_detected = True
    player_x_addr = os.environ.get("PLAYER_X_ADDR")
    progress_scale = float(os.environ.get("PROGRESS_SCALE", "0.05"))
    score_scale = float(os.environ.get("SCORE_SCALE", "0.005"))
    score_clip = float(os.environ.get("SCORE_CLIP", "2.0"))
    time_penalty = float(os.environ.get("TIME_PENALTY", "-0.005"))
    death_penalty = float(os.environ.get("DEATH_PENALTY", "-5.0"))
    frame_skip = int(os.environ.get("FRAME_SKIP", "4"))
    action_hold_s = float(os.environ.get("ACTION_HOLD_S", "0.02"))
    region = CaptureRegion(
        left=region_values["left"],
        top=region_values["top"],
        width=region_values["width"],
        height=region_values["height"],
    )
    # Pixel-based in-game checks are calibrated for a specific capture resolution.
    # When auto-detection changes the region (e.g. container), skip them — RAM-based
    # death detection (credit decrease) is the primary mechanism anyway.
    if auto_detected:
        in_game_checks = []
        continue_checks = []
    else:
        in_game_checks = [
            (400, 10, 160, 220),   # MS6: bright HUD/score area during gameplay
            (200, 30, 80, 140),    # MS6: teal HUD background during gameplay
        ]
        in_game_checks = load_in_game_checks(
            calibration_json_path=calibration_json_path,
            use_calibration=use_calibration,
            default_checks=in_game_checks,
        )
        continue_checks = [
            (640, 340, 170, 255),  # MS6: continue screen center area
        ]

    env = MetalSlugEnv(
        region=region,
        in_game_checks=in_game_checks,
        continue_checks=continue_checks,
        fast_forward_key=fast_forward_key,
        fast_forward_mode=fast_forward_mode,
        fast_forward_state_file=fast_forward_state_file,
        fast_forward_on_reset=fast_forward_on_reset,
        fast_forward_once=fast_forward_once,
        player_x_addr=player_x_addr,
        progress_scale=progress_scale,
        score_scale=score_scale,
        score_clip=score_clip,
        time_penalty=time_penalty,
        death_penalty=death_penalty,
        frame_skip=frame_skip,
        action_hold_s=action_hold_s,
        verbose=verbose,
    )
    # Monitor wraps the env to log episode rewards and lengths
    env = Monitor(
        env,
        filename="outputs/logs/training",
        info_keywords=("phase", "terminate_reason", "score", "score_addr", "episode_reward", "episode_steps"),
    )
    return env


def startup_capture_check(calibration_json_path: str, use_calibration: bool):
    """
    Fail fast when capture appears black/wrong before long PPO training starts.
    """
    region_values = load_region_values(
        calibration_json_path=calibration_json_path,
        use_calibration=use_calibration,
    )
    auto_detected = False
    detected = detect_retroarch_window()
    if detected:
        region_values.update(detected)
        auto_detected = True
    region = CaptureRegion(
        left=region_values["left"],
        top=region_values["top"],
        width=region_values["width"],
        height=region_values["height"],
    )
    if auto_detected:
        in_game_checks = []
    else:
        in_game_checks = [
            (400, 10, 160, 220),
            (200, 30, 80, 140),
        ]
        in_game_checks = load_in_game_checks(
            calibration_json_path=calibration_json_path,
            use_calibration=use_calibration,
            default_checks=in_game_checks,
        )
    probe_env = MetalSlugEnv(
        region=region,
        in_game_checks=in_game_checks,
        continue_checks=[],
        fast_forward_key=None,
        fast_forward_mode="off",
        verbose=0,
    )
    try:
        raw = probe_env._grab_raw()
        frame_min = int(np.min(raw))
        frame_max = int(np.max(raw))
        frame_mean = float(np.mean(raw))
        is_in_game = probe_env._is_in_game(raw) if probe_env.in_game_checks else False
        check_values = probe_env._format_check_values(raw, probe_env.in_game_checks)
        print(
            "[startup-check] frame_stats=min:%s max:%s mean:%.2f in_game=%s checks=%s"
            % (frame_min, frame_max, frame_mean, is_in_game, check_values)
        )
        if frame_max <= 5 or frame_mean < 1.5:
            raise RuntimeError(
                "Startup capture is near-black. Capture region is likely wrong for this display/session. "
                "Check window detection, CAPTURE_LEFT/TOP, and RetroArch video/display settings."
            )
    finally:
        probe_env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100_000,
                        help="Total training timesteps")
    parser.add_argument("--verbose", action="store_true",
                        help="Legacy flag; enables verbose-level >=2")
    parser.add_argument("--verbose-level", type=int, default=int(os.environ.get("VERBOSE_LEVEL", "1")),
                        help="Verbosity level (0-3): 0 quiet, 1 training summary, 2 env+score detail, 3 max debug")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a saved model to resume training from")
    parser.add_argument("--run-until-interrupt", action="store_true",
                        help="Train in chunks until you press Ctrl+C (no step cap); saves after each chunk")
    parser.add_argument("--chunk", type=int, default=50_000,
                        help="Steps per chunk when using --run-until-interrupt (default: 50000)")
    parser.add_argument("--fast-forward-key", type=str, default=None, metavar="KEY",
                        help="Key to press once at start to enable RetroArch fast forward (e.g. tab). Set in RetroArch: Settings -> Input -> Hotkeys -> Fast Forward")
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
    parser.add_argument("--fast-forward-on-reset", action="store_true",
                        help="Send fast-forward key inside env.reset() (recommended for long training sessions)")
    parser.add_argument("--fast-forward-every-reset", action="store_true",
                        help="With --fast-forward-on-reset, send fast-forward key every reset (default sends once)")
    parser.add_argument("--checkpoint-every", type=int, default=10_000, metavar="N",
                        help="Save a checkpoint every N steps (default: 10000). For 200k run, use 20000 to get 10 checkpoints.")
    parser.add_argument("--score-log-every", type=int, default=int(os.environ.get("SCORE_LOG_EVERY", "500")),
                        help="Log monitor metrics every N timesteps (defines one training round)")
    parser.add_argument("--no-score-log-stdout", action="store_true",
                        help="Disable periodic score print lines in stdout")
    parser.add_argument("--target-episodes", type=int, default=int(os.environ.get("TARGET_EPISODES", "0")),
                        help="Stop training after N ended episodes (default: 0 = disabled)")
    parser.add_argument("--video-every-episodes", type=int, default=int(os.environ.get("VIDEO_EVERY_EPISODES", "0")),
                        help="Record a short eval video every N episodes (default: 0 = disabled)")
    parser.add_argument("--video-max-steps", type=int, default=int(os.environ.get("VIDEO_MAX_STEPS", "800")),
                        help="Max steps per recorded video clip (default: 800)")
    parser.add_argument("--video-fps", type=int, default=int(os.environ.get("VIDEO_FPS", "20")),
                        help="FPS for saved video clips (default: 20)")
    parser.add_argument("--video-dir", type=str, default=os.environ.get("VIDEO_DIR", "outputs/videos"),
                        help="Directory for saved video clips (default: outputs/videos)")
    parser.add_argument("--learning-rate", type=float, default=float(os.environ.get("LEARNING_RATE", "3e-4")),
                        help="PPO learning rate (default: 3e-4, env LEARNING_RATE)")
    parser.add_argument("--ent-coef", type=float, default=float(os.environ.get("ENT_COEF", "0.05")),
                        help="PPO entropy coefficient for exploration (default: 0.05, env ENT_COEF)")
    parser.add_argument("--n-steps", type=int, default=int(os.environ.get("N_STEPS", "512")),
                        help="Rollout steps per PPO update (default: 512, env N_STEPS)")
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "64")),
                        help="Mini-batch size for PPO updates (default: 64, env BATCH_SIZE)")
    parser.add_argument("--n-epochs", type=int, default=int(os.environ.get("N_EPOCHS", "4")),
                        help="PPO epochs per update (default: 4, env N_EPOCHS)")
    parser.add_argument("--gamma", type=float, default=float(os.environ.get("GAMMA", "0.99")),
                        help="Discount factor (default: 0.99, env GAMMA)")
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
    parser.add_argument(
        "--skip-startup-capture-check",
        action="store_true",
        help="Skip fail-fast startup frame validation (not recommended)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Training device: 'auto' (CUDA if available, else CPU), 'cuda', or 'cpu'",
    )
    args = parser.parse_args()

    os.makedirs("outputs/logs", exist_ok=True)
    os.makedirs("outputs/models", exist_ok=True)

    in_container = os.environ.get("DISPLAY") and os.environ.get("MSLUG_HEADLESS")
    if not in_container:
        print("Click on RetroArch and load your save state (F4).")
        print("Training starts in 5 seconds...")
        time.sleep(5)
    else:
        print("Headless/container mode: using DISPLAY=%s" % os.environ.get("DISPLAY", ""))

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print("Training device: %s" % device)
    if device == "cuda":
        print("  GPU: %s" % torch.cuda.get_device_name(0))

    fast_forward_mode = args.fast_forward_mode
    if fast_forward_mode is None:
        if args.fast_forward_on_reset:
            fast_forward_mode = "on_reset_every" if args.fast_forward_every_reset else "on_reset_once"
        elif args.fast_forward_key:
            fast_forward_mode = "set_once"
        else:
            fast_forward_mode = "off"
    print("Fast-forward: key=%s mode=%s" % (args.fast_forward_key, fast_forward_mode))

    effective_verbose_level = max(args.verbose_level, 2) if args.verbose else args.verbose_level
    sb3_verbose = 0 if effective_verbose_level <= 0 else (2 if effective_verbose_level >= 3 else 1)
    env_verbose_level = 0 if effective_verbose_level <= 1 else effective_verbose_level

    resolved_region = load_region_values(
        calibration_json_path=args.calibration_json,
        use_calibration=not args.no_calibration,
    )
    detected_region = detect_retroarch_window()
    if detected_region:
        resolved_region.update(detected_region)
    print(
        "Capture region: left=%s top=%s width=%s height=%s (source=%s)"
        % (
            resolved_region["left"],
            resolved_region["top"],
            resolved_region["width"],
            resolved_region["height"],
            resolved_region["source"],
        )
    )
    if not args.skip_startup_capture_check:
        startup_capture_check(
            calibration_json_path=args.calibration_json,
            use_calibration=not args.no_calibration,
        )

    env = DummyVecEnv([
        lambda: make_env(
            verbose=env_verbose_level,
            fast_forward_key=args.fast_forward_key,
            fast_forward_mode=fast_forward_mode,
            fast_forward_state_file=args.fast_forward_state_file,
            fast_forward_on_reset=args.fast_forward_on_reset,
            fast_forward_once=not args.fast_forward_every_reset,
            calibration_json_path=args.calibration_json,
            use_calibration=not args.no_calibration,
        )
    ])

    print("Effective training params:")
    print(
        "  PPO: lr=%s n_steps=%s batch_size=%s n_epochs=%s gamma=%s ent_coef=%s sb3_verbose=%s"
        % (args.learning_rate, args.n_steps, args.batch_size, args.n_epochs, args.gamma, args.ent_coef, sb3_verbose)
    )
    print(
        "  Reward/env: score_scale=%s score_clip=%s progress_scale=%s time_penalty=%s death_penalty=%s player_x_addr=%s capture=(%s,%s,%s,%s) frame_skip=%s action_hold=%s"
        % (
            os.environ.get("SCORE_SCALE", "0.005"),
            os.environ.get("SCORE_CLIP", "2.0"),
            os.environ.get("PROGRESS_SCALE", "0.05"),
            os.environ.get("TIME_PENALTY", "-0.005"),
            os.environ.get("DEATH_PENALTY", "-5.0"),
            os.environ.get("PLAYER_X_ADDR", ""),
            resolved_region["left"],
            resolved_region["top"],
            resolved_region["width"],
            resolved_region["height"],
            os.environ.get("FRAME_SKIP", "4"),
            os.environ.get("ACTION_HOLD_S", "0.02"),
        )
    )
    print(
        "  Episodes/video: target_episodes=%s video_every_episodes=%s video_max_steps=%s video_fps=%s video_dir=%s"
        % (
            args.target_episodes,
            args.video_every_episodes,
            args.video_max_steps,
            args.video_fps,
            args.video_dir,
        )
    )

    # Checkpoints every N steps (and at end). Lets you resume or pick the best.
    checkpoint_cb = CheckpointCallback(
        save_freq=args.checkpoint_every,
        save_path="outputs/models/",
        name_prefix="ppo_mslug6",
    )
    score_cb = ScoreLoggingCallback(
        log_every=args.score_log_every,
        print_stdout=(not args.no_score_log_stdout and effective_verbose_level >= 1),
    )
    episode_cb = EpisodeControlCallback(
        target_episodes=args.target_episodes,
        video_every_episodes=args.video_every_episodes,
        video_max_steps=args.video_max_steps,
        video_fps=args.video_fps,
        video_dir=args.video_dir,
        eval_env_factory=lambda: make_env(
            verbose=0,
            fast_forward_key=None,
            fast_forward_mode="off",
            fast_forward_state_file=args.fast_forward_state_file,
            fast_forward_on_reset=False,
            fast_forward_once=True,
            calibration_json_path=args.calibration_json,
            use_calibration=not args.no_calibration,
        ),
        verbose=effective_verbose_level,
    )
    callbacks = CallbackList([checkpoint_cb, score_cb, episode_cb])

    if args.resume:
        print(f"Resuming from {args.resume}")
        model = PPO.load(args.resume, env=env, device=device)
    else:
        model = PPO(
            "CnnPolicy",
            env,
            verbose=sb3_verbose,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            ent_coef=args.ent_coef,
            tensorboard_log="outputs/ppo_mslug6",
            device=device,
        )

    if args.run_until_interrupt:
        print("Training until you press Ctrl+C (no step cap). Chunk size: %s steps." % args.chunk)
        total_done = 0
        try:
            while True:
                model.learn(
                    total_timesteps=args.chunk,
                    callback=callbacks,
                    reset_num_timesteps=False,
                )
                total_done += args.chunk
                model.save("outputs/models/ppo_mslug6_final")
                print("Chunk done. Total steps so far: %s. Model saved. Continuing..." % total_done)
        except KeyboardInterrupt:
            model.save("outputs/models/ppo_mslug6_final")
            print("\nStopped by user. Total steps: %s. Model saved to outputs/models/ppo_mslug6_final.zip" % total_done)
    else:
        print(f"Training for {args.timesteps} timesteps...")
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
        )
        model.save("outputs/models/ppo_mslug6_final")
        print("Model saved to outputs/models/ppo_mslug6_final.zip")

    env.close()


if __name__ == "__main__":
    main()
