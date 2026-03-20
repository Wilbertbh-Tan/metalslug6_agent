import argparse
import glob
import os
import signal
import shutil
import socket
import sys
import time
from collections import deque

import cv2
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecFrameStack,
    VecNormalize,
)
from src.env.mslug_env import CaptureRegion, MetalSlugEnv
from src.env.region_config import load_in_game_checks, load_region_values
from src.env.window_detect import detect_retroarch_window
from src.impala_cnn import ImpalaCNN


def linear_schedule(initial_value: float):
    """Linear schedule that anneals from initial_value to 0 over training."""

    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return schedule


class ScoreLoggingCallback(BaseCallback):
    """
    Logs score stats during training, including:
    - round average episode score (round = score-log interval)
    - global average episode score
    - global high score and where it happened (episode/step)
    """

    def __init__(
        self,
        log_every: int = 500,
        print_stdout: bool = True,
        stall_threshold: int = 5000,
    ):
        super().__init__(verbose=0)
        self.log_every = max(1, int(log_every))
        self.print_stdout = print_stdout
        self.stall_threshold = stall_threshold
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
        self._rolling_breakdowns = deque(maxlen=100)
        self._last_episode_count_at_check = 0
        self._steps_since_last_episode = 0
        self._stall_warned = False

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
                final_ep_score = (
                    score_value
                    if score_value is not None
                    else float(self.last_score or 0.0)
                )
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
                    deaths = info.get("deaths_this_episode", "?")
                    print(
                        "[episode] ep=%s score=%s reward=%s steps=%s deaths=%s reason=%s"
                        % (
                            self.episode_count,
                            ("%.0f" % final_ep_score),
                            ("%.2f" % ep_reward) if ep_reward is not None else "n/a",
                            ep_steps,
                            deaths,
                            info.get("terminate_reason", "n/a"),
                        )
                    )
                # Per-episode reward breakdown logging
                breakdown = info.get("reward_breakdown")
                if breakdown:
                    self._rolling_breakdowns.append(breakdown)
                    # Log each signal to TensorBoard
                    for key, value in breakdown.items():
                        self.logger.record_mean("reward_breakdown/%s" % key, value)
                    max_x = info.get("max_player_x", 0)
                    self.logger.record_mean(
                        "reward_breakdown/max_player_x", float(max_x or 0)
                    )
                    if self.print_stdout:
                        self._print_breakdown(
                            breakdown,
                            self.episode_count,
                            ep_steps,
                            final_ep_score,
                            deaths,
                            max_x,
                            ep_reward,
                        )

        if self.num_timesteps % self.log_every == 0:
            self.round_index += 1
            round_avg_score = None
            round_avg_reward = None
            if self.round_episode_scores:
                round_avg_score = float(
                    sum(self.round_episode_scores) / len(self.round_episode_scores)
                )
            if self.round_episode_rewards:
                round_avg_reward = float(
                    sum(self.round_episode_rewards) / len(self.round_episode_rewards)
                )
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
            self.logger.record(
                "game/high_score_episode", float(self.high_score_episode)
            )
            if round_avg_score is not None:
                self.logger.record("game/round_avg_episode_score", round_avg_score)
            if global_avg_score is not None:
                self.logger.record("game/global_avg_episode_score", global_avg_score)
            if round_avg_reward is not None:
                self.logger.record("game/round_avg_episode_reward", round_avg_reward)
            if global_avg_reward is not None:
                self.logger.record("game/global_avg_episode_reward", global_avg_reward)

            # Log VecNormalize reward running std (helps diagnose if shaping signals are crushed)
            try:
                vec_env = self.model.get_env()
                if hasattr(vec_env, "ret_rms"):
                    ret_std = float(vec_env.ret_rms.var**0.5)
                    self.logger.record("vecnorm/reward_std", ret_std)
            except Exception:
                pass

            if self.print_stdout:
                print(
                    "[monitor] step=%s eps=%s round_avg_score=%s round_avg_reward=%s "
                    "global_avg_score=%s global_avg_reward=%s high_score=%s"
                    % (
                        self.num_timesteps,
                        self.episode_count,
                        ("%.0f" % round_avg_score)
                        if round_avg_score is not None
                        else "n/a",
                        ("%.2f" % round_avg_reward)
                        if round_avg_reward is not None
                        else "n/a",
                        ("%.0f" % global_avg_score)
                        if global_avg_score is not None
                        else "n/a",
                        ("%.2f" % global_avg_reward)
                        if global_avg_reward is not None
                        else "n/a",
                        ("%.0f" % self.high_score),
                    )
                )
            # Start next round window after logging.
            self.round_episode_scores = []
            self.round_episode_rewards = []

        # Stall detection: flag when no new episodes complete for too long.
        if self.episode_count > self._last_episode_count_at_check:
            self._last_episode_count_at_check = self.episode_count
            self._steps_since_last_episode = 0
            self._stall_warned = False
        else:
            self._steps_since_last_episode += 1
            if (
                self.stall_threshold > 0
                and self._steps_since_last_episode >= self.stall_threshold
                and not self._stall_warned
            ):
                msg = (
                    "[STALL] WARNING: No new episodes completed for %d steps "
                    "(last episode: %d at step %d). Training may be stuck!"
                    % (
                        self._steps_since_last_episode,
                        self.episode_count,
                        self.num_timesteps,
                    )
                )
                print(msg, flush=True)
                self._stall_warned = True
        return True

    def _print_breakdown(
        self, breakdown, ep_num, steps, score, deaths, max_x, ep_reward
    ):
        """Print a formatted reward breakdown table for one episode."""
        # Filter out zero-value signals
        nonzero = {k: v for k, v in breakdown.items() if abs(v) > 1e-9}
        if not nonzero:
            return
        abs_total = sum(abs(v) for v in nonzero.values())
        net = sum(nonzero.values())
        died = "Yes" if deaths and deaths != "?" and int(deaths) > 0 else "No"

        lines = ["  ---- Reward Breakdown (ep %s) ----" % ep_num]
        lines.append(
            "  Steps: %s | Score: %s | Died: %s"
            % (steps, "{:,.0f}".format(score), died)
        )
        if max_x:
            lines.append("  Max Scroll-X: %s" % max_x)

        # Sort by absolute value descending
        for key, value in sorted(
            nonzero.items(), key=lambda x: abs(x[1]), reverse=True
        ):
            pct = abs(value) / abs_total * 100 if abs_total > 0 else 0
            pct_str = "<1%" if 0 < pct < 1 else "%.0f%%" % pct
            sign = "+" if value >= 0 else ""
            lines.append("    %-20s %s%.4f  (%s)" % (key + ":", sign, value, pct_str))

        lines.append("    -----------------------------------")
        lines.append("    %-20s %s%.4f" % ("NET:", "+" if net >= 0 else "", net))

        # Rolling average
        if self._rolling_breakdowns:
            rolling_nets = [sum(b.values()) for b in self._rolling_breakdowns]
            rolling_avg = sum(rolling_nets) / len(rolling_nets)
            lines.append(
                "  Rolling Avg (last %d): %s%.2f"
                % (
                    len(self._rolling_breakdowns),
                    "+" if rolling_avg >= 0 else "",
                    rolling_avg,
                )
            )
        lines.append("  -------------------------------------")
        print("\n".join(lines))


class LatestModelCopyCallback(BaseCallback):
    """Copies new checkpoint files from the run models dir to a shared latest dir."""

    def __init__(self, run_models_dir: str, latest_models_dir: str, run_name: str = ""):
        super().__init__(verbose=0)
        self.run_models_dir = run_models_dir
        self.latest_models_dir = latest_models_dir
        self.run_name = run_name
        self._seen_files = set()

    def _on_step(self) -> bool:
        current_files = set(glob.glob(os.path.join(self.run_models_dir, "*.zip")))
        new_files = current_files - self._seen_files
        for f in new_files:
            basename = os.path.basename(f)
            if self.run_name:
                basename = "%s_%s" % (self.run_name, basename)
            dest = os.path.join(self.latest_models_dir, basename)
            shutil.copy2(f, dest)
        self._seen_files = current_files
        return True


class VecNormalizeSaveCallback(BaseCallback):
    """Saves VecNormalize stats alongside model checkpoints."""

    def __init__(self, save_dir: str, save_freq: int):
        super().__init__(verbose=0)
        self.save_dir = save_dir
        self.save_freq = max(1, save_freq)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            path = os.path.join(self.save_dir, "vec_normalize.pkl")
            self.training_env.save(path)
        return True


class BestModelCallback(BaseCallback):
    """Saves the model when rolling average score (last N episodes) exceeds the best seen."""

    def __init__(self, save_path: str, window: int = 10, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.save_path = save_path
        self.window = max(1, window)
        self._scores: deque = deque(maxlen=window)
        self._best_avg = float("-inf")

    def _on_step(self) -> bool:
        infos = self.locals.get("infos") or []
        dones = self.locals.get("dones")
        if dones is None:
            return True
        done_flags = list(np.asarray(dones).astype(bool))
        for idx, info in enumerate(infos):
            if not info:
                continue
            is_done = done_flags[idx] if idx < len(done_flags) else False
            if not is_done:
                continue
            score = info.get("score")
            if score is not None:
                self._scores.append(float(score))
        if len(self._scores) >= self.window:
            avg = sum(self._scores) / len(self._scores)
            if avg > self._best_avg:
                self._best_avg = avg
                model_path = os.path.join(self.save_path, "ppo_mslug6_best")
                self.model.save(model_path)
                # Also save VecNormalize stats alongside
                vec_norm_path = os.path.join(self.save_path, "vec_normalize_best.pkl")
                self.training_env.save(vec_norm_path)
                print(
                    "[best_model] New best rolling avg score: %.0f (saved to %s)"
                    % (avg, model_path)
                )
        return True


class DeathPenaltyCurriculumCallback(BaseCallback):
    """Gradually increases death penalties as the agent's rolling avg score improves.

    Milestones are (score_threshold, penalty_value) tuples sorted by threshold.
    When rolling avg score exceeds a threshold, penalties are bumped to the next level.
    Uses env_method to update penalties on SubprocVecEnv workers.
    """

    DEFAULT_MILESTONES = [
        (0, -5.0),  # -15 total, breakeven ~8k
        (25000, -8.0),  # -24 total, breakeven ~12k
        (40000, -12.0),  # -36 total, breakeven ~18k
        (50000, -15.0),  # -45 total, breakeven ~23k
        (60000, -18.0),  # -54 total, breakeven ~27k
        (75000, -25.0),  # -75 total, breakeven ~38k
        (90000, -33.0),  # -99 total, breakeven ~50k
        (110000, -42.0),  # -126 total, breakeven ~63k
        (130000, -50.0),  # -150 total, breakeven ~75k
    ]

    def __init__(
        self,
        milestones=None,
        window: int = 100,
        cooldown_episodes: int = 200,
        verbose: int = 1,
    ):
        super().__init__(verbose=0)
        self.milestones = sorted(
            milestones or self.DEFAULT_MILESTONES, key=lambda x: x[0]
        )
        self.window = max(1, window)
        self.cooldown_episodes = max(0, cooldown_episodes)
        self._scores: deque = deque(maxlen=window)
        self._current_level = 0
        self._regression_count = 0
        self._cooldown_remaining = 0
        self._print = verbose >= 1

    def _apply_penalty(self, penalty_value: float):
        """Set death penalties on all underlying envs."""
        venv = self.training_env
        # Walk through wrapper stack to find the base vec env
        while hasattr(venv, "venv"):
            venv = venv.venv
        n_envs = venv.num_envs
        for i in range(n_envs):
            venv.env_method("set_death_penalties", penalty_value, indices=[i])
        if self._print:
            total = penalty_value * 3
            breakeven = abs(total) / 0.002
            print(
                "[curriculum] Death penalty set to %.1f each (%.1f total, breakeven ~%.0fk score)"
                % (penalty_value, total, breakeven / 1000)
            )

    def _on_training_start(self):
        """Apply initial penalty level."""
        _, penalty = self.milestones[self._current_level]
        self._apply_penalty(penalty)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos") or []
        dones = self.locals.get("dones")
        if dones is None:
            return True
        done_flags = list(np.asarray(dones).astype(bool))
        episodes_this_step = 0
        for idx, info in enumerate(infos):
            if not info:
                continue
            is_done = done_flags[idx] if idx < len(done_flags) else False
            if not is_done:
                continue
            episodes_this_step += 1
            score = info.get("score")
            if score is not None:
                self._scores.append(float(score))

        # Decrement cooldown on each completed episode
        if episodes_this_step > 0:
            self._cooldown_remaining = max(
                0, self._cooldown_remaining - episodes_this_step
            )

        # Only check milestones when we have enough data
        if len(self._scores) < self.window:
            return True

        avg = sum(self._scores) / len(self._scores)
        next_level = self._current_level + 1
        if next_level < len(self.milestones) and self._cooldown_remaining <= 0:
            threshold, penalty = self.milestones[next_level]
            if avg >= threshold:
                self._current_level = next_level
                self._apply_penalty(penalty)
                self._regression_count = 0
                if self._print:
                    print(
                        "[curriculum] Rolling avg score %.0f exceeded threshold %d → level %d/%d"
                        % (
                            avg,
                            threshold,
                            self._current_level + 1,
                            len(self.milestones),
                        )
                    )

        # Check for DOWNGRADE — if avg drops below 80% of current level's threshold
        if self._current_level > 0:
            current_threshold, _ = self.milestones[self._current_level]
            downgrade_threshold = current_threshold * 0.8
            if avg < downgrade_threshold:
                self._regression_count += 1
                if self._regression_count >= 100:
                    self._current_level -= 1
                    _, penalty = self.milestones[self._current_level]
                    self._apply_penalty(penalty)
                    self._regression_count = 0
                    self._cooldown_remaining = self.cooldown_episodes
                    if self._print:
                        print(
                            "[curriculum] DOWNGRADE: Rolling avg score %.0f below %.0f → level %d/%d (cooldown %d eps)"
                            % (
                                avg,
                                downgrade_threshold,
                                self._current_level + 1,
                                len(self.milestones),
                                self.cooldown_episodes,
                            )
                        )
            else:
                self._regression_count = 0

        return True


class PlasticityResetCallback(BaseCallback):
    """Periodically reinitialize policy/value linear heads to maintain plasticity.

    Keeps the CNN feature extractor frozen during reset. Addresses the
    loss-of-plasticity problem in continual online RL (Lyle et al., Nature 2024).
    """

    def __init__(self, reset_every: int = 500_000, verbose: int = 1):
        super().__init__(verbose=0)
        self.reset_every = reset_every
        self.print_verbose = verbose
        self._last_reset_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_reset_step >= self.reset_every:
            self._last_reset_step = self.num_timesteps
            policy = self.model.policy
            # Reset action_net and value_net (the linear heads after shared features)
            for name, module in [
                ("action_net", policy.action_net),
                ("value_net", policy.value_net),
            ]:
                if isinstance(module, torch.nn.Linear):
                    torch.nn.init.orthogonal_(
                        module.weight, gain=0.01 if "action" in name else 1.0
                    )
                    torch.nn.init.constant_(module.bias, 0.0)
                    if self.print_verbose >= 1:
                        print(
                            "[plasticity] Reset %s at step %d"
                            % (name, self.num_timesteps)
                        )
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

        filename = "episode_%07d_step_%09d.mp4" % (
            self.episode_count,
            self.num_timesteps,
        )
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
            if (
                self.video_every_episodes > 0
                and self.episode_count % self.video_every_episodes == 0
            ):
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
    monitor_log_dir="outputs/logs",
    reset_via_network=None,
    display=None,
    retroarch_cmd_port=None,
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
    player_x_addr = os.environ.get("PLAYER_X_ADDR", "3FB84E")
    progress_scale = float(os.environ.get("PROGRESS_SCALE", "0.03"))
    score_scale = float(os.environ.get("SCORE_SCALE", "0.002"))
    score_clip = float(os.environ.get("SCORE_CLIP", "2.0"))
    time_penalty = float(os.environ.get("TIME_PENALTY", "-0.002"))
    # HP loss penalties: applied when HP drops during gameplay (before game_mode death)
    hp2 = float(os.environ.get("HP_LOSS_PENALTY_2", "-5.0"))  # HP drops from 2
    hp1 = float(os.environ.get("HP_LOSS_PENALTY_1", "-5.0"))  # HP drops from 1
    hp_loss_penalties = {2: hp2, 1: hp1}
    game_over_penalty = float(os.environ.get("GAME_OVER_PENALTY", "-5.0"))
    # Resource management rewards
    grenade_pickup_reward = float(os.environ.get("GRENADE_PICKUP_REWARD", "0.001"))
    ammo_pickup_reward = float(os.environ.get("AMMO_PICKUP_REWARD", "0.002"))
    grenade_waste_penalty = float(os.environ.get("GRENADE_WASTE_PENALTY", "-0.0001"))
    ammo_waste_penalty = float(os.environ.get("AMMO_WASTE_PENALTY", "-0.00005"))
    score_stall_threshold = int(os.environ.get("SCORE_STALL_THRESHOLD", "60"))
    score_stall_penalty = float(os.environ.get("SCORE_STALL_PENALTY", "-0.002"))
    jump_bonus = float(os.environ.get("JUMP_BONUS", "0.0"))
    jump_bonus_stuck = float(os.environ.get("JUMP_BONUS_STUCK", "0.02"))
    survival_bonus = float(os.environ.get("SURVIVAL_BONUS", "0.003"))
    stuck_threshold_steps = int(os.environ.get("STUCK_THRESHOLD_STEPS", "10"))
    progress_scale_x = float(os.environ.get("PROGRESS_SCALE_X", "0.0"))
    scroll_novelty_bonus = float(os.environ.get("SCROLL_NOVELTY_BONUS", "0.02"))
    scroll_novelty_bucket = int(os.environ.get("SCROLL_NOVELTY_BUCKET", "10"))
    max_episode_steps = int(os.environ.get("MAX_EPISODE_STEPS", "3000"))
    frame_skip = int(os.environ.get("FRAME_SKIP", "3"))
    action_hold_s = float(os.environ.get("ACTION_HOLD_S", "0.005"))
    sticky_action_prob = float(os.environ.get("STICKY_ACTION_PROB", "0.25"))
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
            (400, 10, 160, 220),  # MS6: bright HUD/score area during gameplay
            (200, 30, 80, 140),  # MS6: teal HUD background during gameplay
        ]
        in_game_checks = load_in_game_checks(
            calibration_json_path=calibration_json_path,
            use_calibration=use_calibration,
            default_checks=in_game_checks,
        )
        continue_checks = [
            (640, 340, 170, 255),  # MS6: continue screen center area
        ]

    # In container/headless, use network LOAD_STATE (more reliable than keypress).
    if reset_via_network is None:
        reset_via_network = bool(os.environ.get("MSLUG_HEADLESS"))

    env_kwargs = dict(
        region=region,
        in_game_checks=in_game_checks,
        continue_checks=continue_checks,
        fast_forward_key=fast_forward_key,
        fast_forward_mode=fast_forward_mode,
        fast_forward_state_file=fast_forward_state_file,
        fast_forward_on_reset=fast_forward_on_reset,
        fast_forward_once=fast_forward_once,
        reset_via_network=reset_via_network,
        player_x_addr=player_x_addr,
        progress_scale=progress_scale,
        score_scale=score_scale,
        score_clip=score_clip,
        time_penalty=time_penalty,
        hp_loss_penalties=hp_loss_penalties,
        game_over_penalty=game_over_penalty,
        grenade_pickup_reward=grenade_pickup_reward,
        ammo_pickup_reward=ammo_pickup_reward,
        grenade_waste_penalty=grenade_waste_penalty,
        ammo_waste_penalty=ammo_waste_penalty,
        score_stall_threshold=score_stall_threshold,
        score_stall_penalty=score_stall_penalty,
        jump_bonus=jump_bonus,
        jump_bonus_stuck=jump_bonus_stuck,
        survival_bonus=survival_bonus,
        stuck_threshold_steps=stuck_threshold_steps,
        progress_scale_x=progress_scale_x,
        scroll_novelty_bonus=scroll_novelty_bonus,
        scroll_novelty_bucket=scroll_novelty_bucket,
        max_episode_steps=max_episode_steps,
        sticky_action_prob=sticky_action_prob,
        frame_skip=frame_skip,
        action_hold_s=action_hold_s,
        verbose=verbose,
    )
    if display is not None:
        env_kwargs["display"] = display
    if retroarch_cmd_port is not None:
        env_kwargs["retroarch_cmd_port"] = retroarch_cmd_port
    env = MetalSlugEnv(**env_kwargs)
    # Monitor wraps the env to log episode rewards and lengths
    env = Monitor(
        env,
        filename=os.path.join(monitor_log_dir, "training"),
        info_keywords=(
            "phase",
            "terminate_reason",
            "score",
            "score_addr",
            "episode_reward",
            "episode_steps",
            "deaths_this_episode",
            "lives",
            "bombs",
            "arms",
            "game_time",
            "reward_breakdown",
            "max_player_x",
        ),
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
    parser.add_argument(
        "--timesteps", type=int, default=100_000, help="Total training timesteps"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Legacy flag; enables verbose-level >=2"
    )
    parser.add_argument(
        "--verbose-level",
        type=int,
        default=int(os.environ.get("VERBOSE_LEVEL", "1")),
        help="Verbosity level (0-3): 0 quiet, 1 training summary, 2 env+score detail, 3 max debug",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a saved model to resume training from",
    )
    parser.add_argument(
        "--run-until-interrupt",
        action="store_true",
        help="Train in chunks until you press Ctrl+C (no step cap); saves after each chunk",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=50_000,
        help="Steps per chunk when using --run-until-interrupt (default: 50000)",
    )
    parser.add_argument(
        "--fast-forward-key",
        type=str,
        default=None,
        metavar="KEY",
        help="Key to press once at start to enable RetroArch fast forward (e.g. tab). Set in RetroArch: Settings -> Input -> Hotkeys -> Fast Forward",
    )
    parser.add_argument(
        "--fast-forward-mode",
        type=str,
        choices=[
            "off",
            "set_once",
            "set_once_persist",
            "on_reset_once",
            "on_reset_every",
        ],
        default=None,
        help="Fast-forward behavior (default auto: set_once when key is provided)",
    )
    parser.add_argument(
        "--fast-forward-state-file",
        type=str,
        default="outputs/calibration/fast_forward_state.json",
        help="State file used by set_once_persist mode",
    )
    parser.add_argument(
        "--fast-forward-on-reset",
        action="store_true",
        help="Send fast-forward key inside env.reset() (recommended for long training sessions)",
    )
    parser.add_argument(
        "--fast-forward-every-reset",
        action="store_true",
        help="With --fast-forward-on-reset, send fast-forward key every reset (default sends once)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1_000_000,
        metavar="N",
        help="Save a checkpoint every N steps (default: 1000000). Best model is saved separately on improvement.",
    )
    parser.add_argument(
        "--score-log-every",
        type=int,
        default=int(os.environ.get("SCORE_LOG_EVERY", "500")),
        help="Log monitor metrics every N timesteps (defines one training round)",
    )
    parser.add_argument(
        "--stall-threshold",
        type=int,
        default=int(os.environ.get("STALL_THRESHOLD", "5000")),
        help="Warn if no episodes complete within N steps (default: 5000, 0=disabled)",
    )
    parser.add_argument(
        "--no-score-log-stdout",
        action="store_true",
        help="Disable periodic score print lines in stdout",
    )
    parser.add_argument(
        "--target-episodes",
        type=int,
        default=int(os.environ.get("TARGET_EPISODES", "0")),
        help="Stop training after N ended episodes (default: 0 = disabled)",
    )
    parser.add_argument(
        "--video-every-episodes",
        type=int,
        default=int(os.environ.get("VIDEO_EVERY_EPISODES", "0")),
        help="Record a short eval video every N episodes (default: 0 = disabled)",
    )
    parser.add_argument(
        "--video-max-steps",
        type=int,
        default=int(os.environ.get("VIDEO_MAX_STEPS", "800")),
        help="Max steps per recorded video clip (default: 800)",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=int(os.environ.get("VIDEO_FPS", "20")),
        help="FPS for saved video clips (default: 20)",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default=os.environ.get("VIDEO_DIR", "outputs/videos"),
        help="Directory for saved video clips (default: outputs/videos)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=float(os.environ.get("LEARNING_RATE", "5e-5")),
        help="PPO learning rate (default: 5e-5, env LEARNING_RATE)",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=float(os.environ.get("ENT_COEF", "0.02")),
        help="PPO entropy coefficient for exploration (default: 0.02, env ENT_COEF)",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=int(os.environ.get("N_STEPS", "2048")),
        help="Rollout steps per PPO update (default: 2048, env N_STEPS)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("BATCH_SIZE", "256")),
        help="Mini-batch size for PPO updates (default: 256, env BATCH_SIZE)",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=int(os.environ.get("N_EPOCHS", "4")),
        help="PPO epochs per update (default: 4, env N_EPOCHS)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=float(os.environ.get("GAMMA", "0.99")),
        help="Discount factor (default: 0.99, env GAMMA)",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=float(os.environ.get("GAE_LAMBDA", "0.95")),
        help="GAE lambda for advantage estimation (default: 0.95, env GAE_LAMBDA)",
    )
    parser.add_argument(
        "--clip-range",
        type=float,
        default=float(os.environ.get("CLIP_RANGE", "0.1")),
        help="PPO clip range (default: 0.1, env CLIP_RANGE)",
    )
    parser.add_argument(
        "--clip-range-vf",
        type=float,
        default=float(os.environ.get("CLIP_RANGE_VF", "-1")),
        help="PPO value function clip range (default: -1/disabled, env CLIP_RANGE_VF). Set to -1 to disable.",
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=float(os.environ.get("TARGET_KL", "0.01")),
        help="Target KL divergence for early stopping of PPO updates (default: 0.01, env TARGET_KL). Set to 0 to disable.",
    )
    parser.add_argument(
        "--adam-beta1",
        type=float,
        default=float(os.environ.get("ADAM_BETA1", "0.9")),
        help="Adam optimizer beta1 (default: 0.9, env ADAM_BETA1)",
    )
    parser.add_argument(
        "--adam-beta2",
        type=float,
        default=float(os.environ.get("ADAM_BETA2", "0.9")),
        help="Adam optimizer beta2 (default: 0.9, env ADAM_BETA2). Matched betas prevent policy collapse.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=float(os.environ.get("WEIGHT_DECAY", "1e-4")),
        help="Adam weight decay for plasticity maintenance (default: 1e-4, env WEIGHT_DECAY)",
    )
    parser.add_argument(
        "--use-linear-schedule",
        action="store_true",
        default=os.environ.get("USE_LINEAR_SCHEDULE", "1").lower()
        in ("1", "true", "yes"),
        help="Anneal LR and clip_range linearly to 0 over training (default: on)",
    )
    parser.add_argument(
        "--no-linear-schedule",
        action="store_true",
        help="Disable linear schedule annealing (constant LR and clip_range)",
    )
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
    parser.add_argument(
        "--num-envs",
        type=int,
        default=int(os.environ.get("NUM_ENVS", "1")),
        help="Number of parallel environments (default: 1, env NUM_ENVS). "
        "Each env needs its own Xvfb display and RetroArch instance.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name for this training run. Creates outputs/runs/<run-name>/ with models/, logs/, tensorboard/. "
        "Default: auto-generated from timestamp (e.g. run_20260218_143052)",
    )
    args = parser.parse_args()
    if args.no_linear_schedule:
        args.use_linear_schedule = False

    # Set up run directory
    if args.run_name:
        run_name = args.run_name
    else:
        # Auto-generate PPO_N by finding next available number
        runs_root = os.path.join("outputs", "runs")
        os.makedirs(runs_root, exist_ok=True)
        existing = [
            d for d in os.listdir(runs_root) if d.startswith("PPO_") and d[4:].isdigit()
        ]
        next_num = max((int(d[4:]) for d in existing), default=0) + 1
        run_name = "PPO_%d" % next_num
    run_dir = os.path.join("outputs", "runs", run_name)
    models_dir = os.path.join(run_dir, "models")
    logs_dir = os.path.join(run_dir, "logs")
    tb_dir = os.path.join(run_dir, "tensorboard")
    latest_models_dir = os.path.join("outputs", "models")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(latest_models_dir, exist_ok=True)

    # Tee stdout to a log file so training output persists outside the container.
    log_path = os.path.join(logs_dir, "training_stdout.log")

    class TeeWriter:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()

        def flush(self):
            for s in self.streams:
                s.flush()

    log_file = open(log_path, "a")
    sys.stdout = TeeWriter(sys.__stdout__, log_file)
    sys.stderr = TeeWriter(sys.__stderr__, log_file)
    print("\n=== Training started at %s ===" % time.strftime("%Y-%m-%d %H:%M:%S"))
    print("Run: %s" % run_dir)

    in_container = os.environ.get("DISPLAY") and os.environ.get("MSLUG_HEADLESS")
    if not in_container:
        print("Click on RetroArch and load your save state (F4).")
        print("Training starts in 5 seconds...")
        time.sleep(5)
    else:
        print(
            "Headless/container mode: using DISPLAY=%s" % os.environ.get("DISPLAY", "")
        )

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
            fast_forward_mode = (
                "on_reset_every" if args.fast_forward_every_reset else "on_reset_once"
            )
        elif args.fast_forward_key:
            fast_forward_mode = "set_once"
        else:
            fast_forward_mode = "off"
    print("Fast-forward: key=%s mode=%s" % (args.fast_forward_key, fast_forward_mode))

    effective_verbose_level = (
        max(args.verbose_level, 2) if args.verbose else args.verbose_level
    )
    sb3_verbose = (
        0
        if effective_verbose_level <= 0
        else (2 if effective_verbose_level >= 3 else 1)
    )
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

    def make_env_fn(env_index: int):
        """Factory that returns a closure for creating env with per-worker display/port."""
        env_display = ":%d" % (99 + env_index)
        env_cmd_port = 55355 + env_index

        def _init():
            return make_env(
                verbose=env_verbose_level,
                fast_forward_key=args.fast_forward_key,
                fast_forward_mode=fast_forward_mode,
                fast_forward_state_file=args.fast_forward_state_file,
                fast_forward_on_reset=args.fast_forward_on_reset,
                fast_forward_once=not args.fast_forward_every_reset,
                calibration_json_path=args.calibration_json,
                use_calibration=not args.no_calibration,
                monitor_log_dir=logs_dir,
                display=env_display if args.num_envs > 1 else None,
                retroarch_cmd_port=env_cmd_port if args.num_envs > 1 else None,
            )

        return _init

    n_envs = args.num_envs

    def build_vec_env():
        """Create SubprocVecEnv/DummyVecEnv + VecFrameStack (no VecNormalize)."""
        if n_envs > 1:
            venv = SubprocVecEnv([make_env_fn(i) for i in range(n_envs)])
            print("Using SubprocVecEnv with %d parallel environments" % n_envs)
        else:
            venv = DummyVecEnv([make_env_fn(0)])
        return VecFrameStack(venv, n_stack=4)

    env = build_vec_env()
    print(
        "Wrappers: VecFrameStack(4) + VecNormalize(norm_obs=False, norm_reward=True, clip_reward=10)"
    )
    print("CNN: ImpalaCNN (ResNet 32->64->64 + GAP, features_dim=512, DrAC augment)")

    print("Effective training params:")
    lr_display = (
        "lin_schedule(%s)" % args.learning_rate
        if args.use_linear_schedule
        else str(args.learning_rate)
    )
    clip_display = (
        "lin_schedule(%s)" % args.clip_range
        if args.use_linear_schedule
        else str(args.clip_range)
    )
    clip_vf_display = "None" if args.clip_range_vf < 0 else str(args.clip_range_vf)
    target_kl_display = str(args.target_kl) if args.target_kl > 0 else "None"
    print(
        "  PPO: lr=%s clip_range=%s clip_range_vf=%s target_kl=%s n_steps=%s batch_size=%s n_epochs=%s "
        "gamma=%s gae_lambda=%s ent_coef=%s num_envs=%s adam_betas=(%s,%s) sb3_verbose=%s"
        % (
            lr_display,
            clip_display,
            clip_vf_display,
            target_kl_display,
            args.n_steps,
            args.batch_size,
            args.n_epochs,
            args.gamma,
            args.gae_lambda,
            args.ent_coef,
            args.num_envs,
            args.adam_beta1,
            args.adam_beta2,
            sb3_verbose,
        )
    )
    print(
        "  Reward/env: score_scale=%s score_clip=%s progress_scale=%s time_penalty=%s hp_loss={2:%s,1:%s} game_over=%s survival_bonus=%s"
        % (
            os.environ.get("SCORE_SCALE", "0.002"),
            os.environ.get("SCORE_CLIP", "2.0"),
            os.environ.get("PROGRESS_SCALE", "0.03"),
            os.environ.get("TIME_PENALTY", "-0.002"),
            os.environ.get("HP_LOSS_PENALTY_2", "-5.0"),
            os.environ.get("HP_LOSS_PENALTY_1", "-5.0"),
            os.environ.get("GAME_OVER_PENALTY", "-5.0"),
            os.environ.get("SURVIVAL_BONUS", "0.003"),
        )
    )
    print(
        "  Resources: grenade_pickup=%s ammo_pickup=%s grenade_waste=%s ammo_waste=%s"
        % (
            os.environ.get("GRENADE_PICKUP_REWARD", "0.001"),
            os.environ.get("AMMO_PICKUP_REWARD", "0.002"),
            os.environ.get("GRENADE_WASTE_PENALTY", "-0.0001"),
            os.environ.get("AMMO_WASTE_PENALTY", "-0.00005"),
        )
    )
    print(
        "  Stall/explore: score_stall_threshold=%s score_stall_penalty=%s jump_bonus=%s "
        "jump_bonus_stuck=%s stuck_threshold=%s progress_scale_x=%s max_episode_steps=%s"
        % (
            os.environ.get("SCORE_STALL_THRESHOLD", "60"),
            os.environ.get("SCORE_STALL_PENALTY", "-0.002"),
            os.environ.get("JUMP_BONUS", "0.0"),
            os.environ.get("JUMP_BONUS_STUCK", "0.02"),
            os.environ.get("STUCK_THRESHOLD_STEPS", "10"),
            os.environ.get("PROGRESS_SCALE_X", "0.0"),
            os.environ.get("MAX_EPISODE_STEPS", "3000"),
        )
    )
    print(
        "  Env: player_x_addr=%s capture=(%s,%s,%s,%s) frame_skip=%s action_hold=%s sticky_prob=%s"
        % (
            os.environ.get("PLAYER_X_ADDR", "3FB84E"),
            resolved_region["left"],
            resolved_region["top"],
            resolved_region["width"],
            resolved_region["height"],
            os.environ.get("FRAME_SKIP", "3"),
            os.environ.get("ACTION_HOLD_S", "0.005"),
            os.environ.get("STICKY_ACTION_PROB", "0.25"),
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

    print(
        "  Death penalty curriculum: %s"
        % " → ".join(
            "%.0fk:%.0f" % (t / 1000, p)
            for t, p in DeathPenaltyCurriculumCallback.DEFAULT_MILESTONES
        )
    )

    # Checkpoints every N steps (and at end). Lets you resume or pick the best.
    checkpoint_cb = CheckpointCallback(
        save_freq=args.checkpoint_every,
        save_path=models_dir,
        name_prefix="ppo_mslug6",
    )
    score_cb = ScoreLoggingCallback(
        log_every=args.score_log_every,
        print_stdout=(not args.no_score_log_stdout and effective_verbose_level >= 1),
        stall_threshold=args.stall_threshold,
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
            monitor_log_dir=logs_dir,
        ),
        verbose=effective_verbose_level,
    )
    latest_copy_cb = LatestModelCopyCallback(
        models_dir, latest_models_dir, run_name=run_name
    )
    vec_norm_cb = VecNormalizeSaveCallback(models_dir, save_freq=args.checkpoint_every)
    best_model_cb = BestModelCallback(save_path=models_dir, window=10)
    curriculum_cb = DeathPenaltyCurriculumCallback(window=100, verbose=1)
    callbacks = CallbackList(
        [
            checkpoint_cb,
            latest_copy_cb,
            vec_norm_cb,
            best_model_cb,
            score_cb,
            episode_cb,
            curriculum_cb,
        ]
    )

    # Apply linear schedule annealing for LR and clip_range (Atari standard)
    lr = (
        linear_schedule(args.learning_rate)
        if args.use_linear_schedule
        else args.learning_rate
    )
    clip_range = (
        linear_schedule(args.clip_range)
        if args.use_linear_schedule
        else args.clip_range
    )
    clip_range_vf = None if args.clip_range_vf < 0 else args.clip_range_vf
    target_kl = args.target_kl if args.target_kl > 0 else None
    optimizer_kwargs = {
        "betas": (args.adam_beta1, args.adam_beta2),
        "weight_decay": args.weight_decay,
    }

    if args.resume:
        print(f"Resuming from {args.resume}")
        # Load VecNormalize stats if saved alongside the model, otherwise create fresh
        model_dir = os.path.dirname(args.resume)
        vec_norm_best = os.path.join(model_dir, "vec_normalize_best.pkl")
        vec_norm_default = os.path.join(model_dir, "vec_normalize.pkl")
        vec_norm_path = (
            vec_norm_best if os.path.exists(vec_norm_best) else vec_norm_default
        )
        if os.path.exists(vec_norm_path):
            env = VecNormalize.load(vec_norm_path, env)
            env.training = True
            print(f"  Loaded VecNormalize stats from {vec_norm_path}")
        else:
            env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)
        model = PPO.load(
            args.resume,
            env=env,
            device=device,
            learning_rate=lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            ent_coef=args.ent_coef,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            target_kl=target_kl,
        )
        # Override Adam betas on resume (optimizer state is rebuilt anyway)
        model.policy.optimizer_kwargs = optimizer_kwargs
    else:
        env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)
        policy_kwargs = {
            "optimizer_kwargs": optimizer_kwargs,
            "features_extractor_class": ImpalaCNN,
            "features_extractor_kwargs": {"features_dim": 512},
        }
        model = PPO(
            "CnnPolicy",
            env,
            verbose=sb3_verbose,
            learning_rate=lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            ent_coef=args.ent_coef,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            target_kl=target_kl,
            tensorboard_log=tb_dir,
            device=device,
            policy_kwargs=policy_kwargs,
        )

    def _save_model_and_stats():
        """Save model + VecNormalize stats + copy to latest."""
        model.save(os.path.join(models_dir, "ppo_mslug6_final"))
        env.save(os.path.join(models_dir, "vec_normalize.pkl"))
        shutil.copy2(
            os.path.join(models_dir, "ppo_mslug6_final.zip"),
            os.path.join(latest_models_dir, "%s_ppo_mslug6_final.zip" % run_name),
        )
        shutil.copy2(
            os.path.join(models_dir, "vec_normalize.pkl"),
            os.path.join(latest_models_dir, "%s_vec_normalize.pkl" % run_name),
        )

    crash_checkpoint_path = os.path.join(models_dir, "ppo_mslug6_crash_recovery")
    crash_vec_norm_path = os.path.join(models_dir, "vec_normalize_crash_recovery.pkl")
    crash_count = 0
    _CRASH_ERRORS = (EOFError, ConnectionResetError, BrokenPipeError)

    def _recover_from_crash():
        """Save current state, rebuild env, reload model, restore curriculum."""
        nonlocal env, model, crash_count, callbacks
        crash_count += 1
        print(
            "[crash-recovery] Worker crashed (crash #%d), saving and rebuilding..."
            % crash_count
        )
        # Save current model + VecNormalize stats
        try:
            model.save(crash_checkpoint_path)
            print("[crash-recovery] Model saved to %s" % crash_checkpoint_path)
        except Exception as e:
            print("[crash-recovery] Warning: could not save model: %s" % e)
        try:
            env.save(crash_vec_norm_path)
            print(
                "[crash-recovery] VecNormalize stats saved to %s" % crash_vec_norm_path
            )
        except Exception as e:
            print("[crash-recovery] Warning: could not save VecNormalize: %s" % e)
        # Close broken env (may raise)
        try:
            env.close()
        except Exception:
            pass
        # Check RetroArch health on each port before rebuilding
        for i in range(n_envs):
            port = 55355 + i
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.settimeout(1.0)
                sock.sendto(b"GET_STATUS\n", ("127.0.0.1", port))
                sock.recvfrom(4096)
                sock.close()
                print("[crash-recovery] RetroArch on port %d is responsive" % port)
            except (socket.timeout, OSError):
                sock.close()
                print("[crash-recovery] RetroArch on port %d is NOT responding" % port)
                print(
                    "[crash-recovery] FATAL: cannot rebuild env — RetroArch is dead. "
                    "Checkpoint saved at %s" % crash_checkpoint_path
                )
                print(
                    "[crash-recovery] Restart container to resume from crash checkpoint."
                )
                sys.exit(1)

        # Rebuild env from scratch (with timeout to avoid infinite hang)
        def _alarm_handler(signum, frame):
            raise TimeoutError("build_vec_env timed out")

        old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(120)
        try:
            new_env = build_vec_env()
            signal.alarm(0)
        except TimeoutError:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            print(
                "[crash-recovery] FATAL: env rebuild timed out after 120s. "
                "Checkpoint saved at %s" % crash_checkpoint_path
            )
            print("[crash-recovery] Restart container to resume from crash checkpoint.")
            sys.exit(1)
        signal.signal(signal.SIGALRM, old_handler)
        if os.path.exists(crash_vec_norm_path):
            new_env = VecNormalize.load(crash_vec_norm_path, new_env)
            new_env.training = True
        else:
            new_env = VecNormalize(
                new_env, norm_obs=False, norm_reward=True, clip_reward=10.0
            )
        env = new_env
        # Reload model with new env
        if os.path.exists(crash_checkpoint_path + ".zip"):
            model = PPO.load(
                crash_checkpoint_path,
                env=env,
                device=device,
                learning_rate=lr,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                ent_coef=args.ent_coef,
                clip_range=clip_range,
                clip_range_vf=clip_range_vf,
                target_kl=target_kl,
            )
            model.policy.optimizer_kwargs = optimizer_kwargs
        # Restore curriculum callback state — re-apply current penalty level
        curriculum_cb._apply_penalty(
            curriculum_cb.milestones[curriculum_cb._current_level][1]
        )
        # Rebuild callback list so all callbacks point to new model/env
        callbacks = CallbackList(
            [
                checkpoint_cb,
                latest_copy_cb,
                vec_norm_cb,
                best_model_cb,
                score_cb,
                episode_cb,
                curriculum_cb,
            ]
        )
        print(
            "[crash-recovery] Recovered (crash #%d), continuing training..."
            % crash_count
        )

    if args.run_until_interrupt:
        print(
            "Training until you press Ctrl+C (no step cap). Chunk size: %s steps."
            % args.chunk
        )
        total_done = 0
        try:
            while True:
                try:
                    model.learn(
                        total_timesteps=args.chunk,
                        callback=callbacks,
                        reset_num_timesteps=False,
                    )
                    total_done += args.chunk
                    _save_model_and_stats()
                    print(
                        "Chunk done. Total steps so far: %s. Model saved. Continuing..."
                        % total_done
                    )
                except _CRASH_ERRORS as e:
                    print("[crash-recovery] Caught %s: %s" % (type(e).__name__, e))
                    _recover_from_crash()
                    total_done = model.num_timesteps
        except KeyboardInterrupt:
            _save_model_and_stats()
            print(
                "\nStopped by user. Total steps: %s. Model saved to %s/ppo_mslug6_final.zip"
                % (total_done, models_dir)
            )
    else:
        print(f"Training for {args.timesteps} timesteps...")
        timesteps_remaining = args.timesteps
        while timesteps_remaining > 0:
            try:
                model.learn(
                    total_timesteps=timesteps_remaining,
                    callback=callbacks,
                    reset_num_timesteps=(crash_count == 0),
                )
                break  # completed successfully
            except _CRASH_ERRORS as e:
                print("[crash-recovery] Caught %s: %s" % (type(e).__name__, e))
                _recover_from_crash()
                timesteps_remaining = args.timesteps - model.num_timesteps
                if timesteps_remaining <= 0:
                    break
                print("[crash-recovery] %d timesteps remaining" % timesteps_remaining)
        _save_model_and_stats()
        print("Model saved to %s/ppo_mslug6_final.zip" % models_dir)

    env.close()


if __name__ == "__main__":
    main()
