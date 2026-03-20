"""
Metal Slug gymnasium environment.

Action space: MultiDiscrete([5, 3, 3]) — movement, attack, and modifier
can be combined simultaneously (e.g. right+shoot+jump).

Game over detection: RAM-based via game_mode at 003868DA.
game_mode=1 means alive, game_mode>1 means dead.
"""
import json
import os
import signal
import socket
import subprocess
import time
from dataclasses import dataclass

import cv2
import gymnasium as gym
import mss
import numpy as np
import pyautogui

from src.env.ram_decode import decode_bcd_score
from src.env.rewards import compute_reward


def _kill_stale_agents():
    """Kill zombie agent/monitor processes from previous runs.

    Called automatically when MetalSlugEnv is created to prevent
    UDP socket cross-contamination from stale processes.
    """
    my_pid = os.getpid()
    patterns = ["random_agent", "eval_ppo", "reward_monitor", "train_ppo"]
    killed = []
    for pat in patterns:
        try:
            result = subprocess.run(
                ["pgrep", "-f", pat],
                capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.strip().split("\n"):
                pid = line.strip()
                if pid and int(pid) != my_pid:
                    try:
                        os.kill(int(pid), signal.SIGKILL)
                        killed.append((pid, pat))
                    except ProcessLookupError:
                        pass
        except (subprocess.TimeoutExpired, ValueError):
            continue
    # Also kill inline python -c reward monitors
    try:
        result = subprocess.run(
            ["pgrep", "-f", "READ_CORE_RAM"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.strip().split("\n"):
            pid = line.strip()
            if pid and int(pid) != my_pid:
                try:
                    os.kill(int(pid), signal.SIGKILL)
                    killed.append((pid, "inline_monitor"))
                except ProcessLookupError:
                    pass
    except (subprocess.TimeoutExpired, ValueError):
        pass
    if killed:
        for pid, pat in killed:
            print(f"  Killed stale process {pid} ({pat})")
    return killed


@dataclass
class CaptureRegion:
    left: int
    top: int
    width: int = 640
    height: int = 480


class MetalSlugEnv(gym.Env):
    def __init__(
        self,
        region: CaptureRegion | None = None,
        action_hold_s: float = 0.02,
        frame_skip: int = 4,
        shoot_key: str = "z",
        jump_key: str = "x",
        grenade_key: str = "a",
        metal_slug_attack_key: str = "s",
        weapon_change_key: str = "w",
        load_state_key: str = "f4",
        fast_forward_key: str | None = None,
        fast_forward_mode: str = "off",
        fast_forward_state_file: str = "outputs/calibration/fast_forward_state.json",
        fast_forward_on_reset: bool = False,
        fast_forward_once: bool = True,
        game_over_checks: list[tuple[int, int, int, int]] | None = None,
        continue_checks: list[tuple[int, int, int, int]] | None = None,
        in_game_checks: list[tuple[int, int, int, int]] | None = None,
        enforce_in_game_checks: bool = True,
        reset_wait_for_in_game_s: float = 10.0,
        reset_check_interval_s: float = 0.2,
        max_transition_steps: int = 60,
        reset_via_network: bool = False,
        retroarch_host: str = "127.0.0.1",
        retroarch_cmd_port: int = 55355,
        ram_read_retries: int = 3,
        player_x_addr: str | None = None,
        lives_addr: str = "003868D1",
        progress_scale: float = 0.05,
        score_scale: float = 0.005,
        score_clip: float = 2.0,
        time_penalty: float = -0.005,
        death_penalty: float = -5.0,
        verbose: bool | int = False,
    ):
        super().__init__()
        _kill_stale_agents()
        self.region = region or CaptureRegion(left=1116, top=345)
        self.action_hold_s = action_hold_s
        self.frame_skip = max(1, int(frame_skip))
        self.shoot_key = shoot_key
        self.jump_key = jump_key
        self.grenade_key = grenade_key
        self.metal_slug_attack_key = metal_slug_attack_key
        self.weapon_change_key = weapon_change_key
        self.load_state_key = load_state_key
        self.fast_forward_key = fast_forward_key
        # Fast-forward behavior:
        # - off: never send fast-forward key
        # - set_once: send key on first reset only (recommended for toggle hotkeys)
        # - set_once_persist: send once total across runs, tracked in a state file
        # - on_reset_once: send on first reset only (legacy-compatible alias)
        # - on_reset_every: send every reset
        mode = str(fast_forward_mode or "off").strip().lower()
        valid_modes = {"off", "set_once", "set_once_persist", "on_reset_once", "on_reset_every"}
        if mode not in valid_modes:
            raise ValueError(f"Invalid fast_forward_mode={fast_forward_mode!r}. Expected one of {sorted(valid_modes)}")
        # Backward compatibility with legacy flags.
        if mode == "off":
            if fast_forward_on_reset:
                mode = "on_reset_once" if fast_forward_once else "on_reset_every"
            elif fast_forward_key:
                # Safe default when a key exists: enable once and do not re-toggle.
                mode = "set_once"
        self.fast_forward_mode = mode
        self.fast_forward_state_file = fast_forward_state_file
        self._fast_forward_armed = True
        self.game_over_checks = game_over_checks or []
        self.continue_checks = continue_checks or []
        self.in_game_checks = in_game_checks or []
        self.enforce_in_game_checks = enforce_in_game_checks
        self.reset_wait_for_in_game_s = reset_wait_for_in_game_s
        self.reset_check_interval_s = reset_check_interval_s
        self.max_transition_steps = max_transition_steps
        self.reset_via_network = reset_via_network
        self.retroarch_host = retroarch_host
        self.retroarch_cmd_port = int(retroarch_cmd_port)
        self.ram_read_retries = max(1, int(ram_read_retries))
        self.player_x_addr = player_x_addr
        self.lives_addr = lives_addr
        self._game_state_addr = "003868D0"
        self.progress_scale = progress_scale
        self.score_scale = score_scale
        self.score_clip = score_clip
        self.time_penalty = time_penalty
        self.death_penalty = death_penalty
        if isinstance(verbose, bool):
            self.verbose_level = 1 if verbose else 0
        else:
            self.verbose_level = int(verbose)
        self._prev_x: int | None = None
        # MultiDiscrete([5, 3, 3]):
        #   dim 0 — movement:  0=noop, 1=left, 2=right, 3=up, 4=down
        #   dim 1 — attack:    0=noop, 1=shoot, 2=grenade
        #   dim 2 — modifier:  0=noop, 1=jump, 2=slug_attack
        self._movement_names = ["noop", "left", "right", "up", "down"]
        self._attack_names = ["noop", "shoot", "grenade"]
        self._modifier_names = ["noop", "jump", "slug_atk"]
        self._prev_score = 0
        self._step_count = 0
        self._episode_count = 0
        self._consecutive_transition = 0
        self._last_end_reason = "startup"
        self._episode_reward = 0.0
        self._seen_alive = False  # track if game_mode==1 seen since last reset
        self._score_addr = "003869BC"
        self._ram_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._ram_sock.settimeout(0.1)
        # Dedicated socket for status block reads (death detection) to avoid
        # cross-contamination with other RAM reads sharing _ram_sock.
        self._status_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._status_sock.settimeout(0.1)
        self._cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._cmd_sock.settimeout(0.1)
        self.action_space = gym.spaces.MultiDiscrete([5, 3, 3])
        self.observation_space = gym.spaces.Box(0, 255, shape=(84, 84, 1), dtype=np.uint8)
        self.sct = mss.mss()
        pyautogui.FAILSAFE = True

    def _grab_raw(self):
        frame = np.array(self.sct.grab(self.region.__dict__))
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)

    def render(self):
        """
        Return current frame as RGB array for video/debug capture.
        """
        frame = np.array(self.sct.grab(self.region.__dict__))
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

    def _preprocess(self, gray_frame):
        frame = cv2.resize(gray_frame, (84, 84))
        return frame[..., None]

    def _tap_key(self, key: str):
        pyautogui.keyDown(key)
        time.sleep(self.action_hold_s)
        pyautogui.keyUp(key)

    def _matches_pixel_checks(self, gray_frame, checks):
        for x, y, min_val, max_val in checks:
            val = int(gray_frame[y, x])
            if val < min_val or val > max_val:
                return False
        return True

    def _is_in_game(self, gray_frame):
        return bool(self.in_game_checks) and self._matches_pixel_checks(gray_frame, self.in_game_checks)

    def _read_status_block(self) -> tuple[int | None, int | None]:
        """Read game_state and game_mode from RAM.

        Addresses 003868D0 (game_state) and 003868DA (game_mode).
        game_mode=1 means alive/playing, game_mode>1 means dead.
        Returns (game_state, game_mode) or (None, None).
        """
        for _ in range(self.ram_read_retries):
            try:
                cmd = "READ_CORE_RAM 003868D0 11\n"
                self._status_sock.sendto(cmd.encode(), (self.retroarch_host, self.retroarch_cmd_port))
                data, _ = self._status_sock.recvfrom(1024)
                parts = data.decode().strip().split()
                values = [int(b, 16) for b in parts[2:]]
                if len(values) == 11:
                    return values[0], values[10]
            except socket.timeout:
                continue
        return None, None

    def _is_ram_death(self) -> bool:
        """Detect death via game_mode at 003868DA.

        game_mode=1 means alive/playing. game_mode>1 means the player
        is dead (death animation, continue screen, game over).

        After reset, game_mode may not settle to 1 immediately. We wait
        until we've seen game_mode==1 at least once before triggering
        death detection, to avoid false positives from save state load.
        """
        gs, game_mode = self._read_status_block()
        self._log(2, f"[death_check] gs={gs} game_mode={game_mode} seen_alive={self._seen_alive}")
        if game_mode is None:
            return False
        if not self._seen_alive:
            if game_mode == 1:
                self._seen_alive = True
            return False
        return game_mode > 1

    def _read_ram(self, address, num_bytes=4):
        for _ in range(self.ram_read_retries):
            try:
                cmd = f"READ_CORE_RAM {address} {num_bytes}\n"
                self._ram_sock.sendto(cmd.encode(), (self.retroarch_host, self.retroarch_cmd_port))
                data, _ = self._ram_sock.recvfrom(1024)
                parts = data.decode().strip().split()
                values = [int(b, 16) for b in parts[2:]]
                if len(values) == num_bytes:
                    return values
            except socket.timeout:
                continue
        return []

    def _send_retroarch_cmd(self, command: str):
        cmd = command.strip().upper() + "\n"
        self._cmd_sock.sendto(cmd.encode(), (self.retroarch_host, self.retroarch_cmd_port))

    def _read_score(self):
        b = self._read_ram(self._score_addr)
        decoded = decode_bcd_score(b)
        if decoded is None:
            return self._prev_score
        return decoded

    def _read_player_x(self) -> int | None:
        """Read player horizontal position from RAM. Supports 16-bit LE int or 32-bit LE float.

        When player_x_addr ends with ':f32', reads as IEEE 754 float and rounds to int.
        Otherwise reads as 16-bit little-endian unsigned integer.
        """
        if not self.player_x_addr:
            return None
        addr = self.player_x_addr
        if addr.endswith(":f32"):
            addr = addr[:-4]
            b = self._read_ram(addr, 4)
            if not b or len(b) < 4:
                return None
            import struct
            val = struct.unpack("<f", bytes(b))[0]
            if val != val:  # NaN check
                return None
            return int(val)
        b = self._read_ram(addr, 2)
        if not b or len(b) < 2:
            return None
        return b[0] | (b[1] << 8)

    def _log(self, level: int, msg: str):
        if self.verbose_level >= level:
            print(msg)

    def _maybe_enable_fast_forward(self):
        if not self.fast_forward_key:
            return
        if self.fast_forward_mode == "off":
            return
        if self.fast_forward_mode == "set_once_persist":
            if self._is_fast_forward_marked_enabled():
                self._log(1, f"[reset] Fast-forward already marked enabled, skipping key ({self.fast_forward_key})")
                return
        if self.fast_forward_mode in ("set_once", "on_reset_once") and not self._fast_forward_armed:
            return
        self._tap_key(self.fast_forward_key)
        if self.fast_forward_mode in ("set_once", "on_reset_once", "set_once_persist"):
            self._fast_forward_armed = False
        if self.fast_forward_mode == "set_once_persist":
            self._mark_fast_forward_enabled()
        self._log(
            1,
            f"[reset] Fast-forward key sent ({self.fast_forward_key}) mode={self.fast_forward_mode}",
        )

    def _is_fast_forward_marked_enabled(self) -> bool:
        try:
            if not self.fast_forward_state_file or not os.path.exists(self.fast_forward_state_file):
                return False
            with open(self.fast_forward_state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            return bool(payload.get("enabled")) and payload.get("key") == self.fast_forward_key
        except Exception:
            return False

    def _mark_fast_forward_enabled(self):
        try:
            state_dir = os.path.dirname(self.fast_forward_state_file)
            if state_dir:
                os.makedirs(state_dir, exist_ok=True)
            payload = {
                "enabled": True,
                "key": self.fast_forward_key,
                "updated_at": time.time(),
            }
            with open(self.fast_forward_state_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            # Do not break training/eval if state write fails.
            pass

    def _format_check_values(self, gray_frame, checks) -> str:
        if not checks:
            return "[]"
        parts = []
        for x, y, min_val, max_val in checks:
            val = int(gray_frame[y, x])
            parts.append(f"({x},{y})={val} in [{min_val},{max_val}]")
        return "[" + ", ".join(parts) + "]"

    def _compute_reward(self, action):
        curr_score = self._read_score()
        reward, info = compute_reward(
            action=action,
            curr_score=curr_score,
            prev_score=self._prev_score,
            score_scale=self.score_scale,
            score_clip=self.score_clip,
            progress_scale=self.progress_scale,
            time_penalty=self.time_penalty,
        )
        self._prev_score = curr_score
        return reward, info

    def _format_action(self, action):
        """Format a MultiDiscrete action as a readable string."""
        mv = self._movement_names[action[0]]
        atk = self._attack_names[action[1]]
        mod = self._modifier_names[action[2]]
        parts = [p for p in [mv, atk, mod] if p != "noop"]
        return "+".join(parts) or "noop"

    def _execute_action(self, action):
        """Press multiple keys simultaneously based on MultiDiscrete action."""
        movement, attack, modifier = action[0], action[1], action[2]
        keys = []
        if movement == 1:
            keys.append("left")
        elif movement == 2:
            keys.append("right")
        elif movement == 3:
            keys.append("up")
        elif movement == 4:
            keys.append("down")

        if attack == 1:
            keys.append(self.shoot_key)
        elif attack == 2:
            keys.append(self.grenade_key)

        if modifier == 1:
            keys.append(self.jump_key)
        elif modifier == 2:
            keys.append(self.metal_slug_attack_key)

        if not keys:
            time.sleep(self.action_hold_s)
            return

        for k in keys:
            pyautogui.keyDown(k)
        time.sleep(self.action_hold_s)
        for k in keys:
            pyautogui.keyUp(k)

    def step(self, action):
        # SB3 passes MultiDiscrete as numpy array; ensure flat int array.
        action = np.asarray(action).flatten().astype(int)

        # Frame skip: repeat the action multiple times, checking for death
        # each iteration but only computing full reward on the last frame.
        for skip_i in range(self.frame_skip):
            raw = self._grab_raw()
            last_frame = (skip_i == self.frame_skip - 1)

            terminated = self._is_ram_death()
            if terminated:
                self._last_end_reason = "ram_death"
                obs = self._preprocess(raw)
                curr_score = self._read_score()
                self._episode_reward += self.death_penalty
                self._log(1, f"[step {self._step_count}] DEATH detected, reward={self.death_penalty}")
                return obs, self.death_penalty, True, False, {
                    "phase": "game_over",
                    "terminate_reason": "ram_death",
                    "score": curr_score,
                    "score_addr": self._score_addr,
                    "episode_reward": self._episode_reward,
                    "episode_steps": self._step_count,
                }

            if self.enforce_in_game_checks and self.in_game_checks and not self._is_in_game(raw):
                self._consecutive_transition += 1
                obs = self._preprocess(raw)
                curr_score = self._read_score()
                self._log(1, f"[step {self._step_count}] TRANSITION [{self._consecutive_transition}/{self.max_transition_steps}]")
                if self.max_transition_steps > 0 and self._consecutive_transition >= self.max_transition_steps:
                    self._last_end_reason = "transition_timeout"
                    self._episode_reward += self.death_penalty
                    self._log(1, f"[step {self._step_count}] TRANSITION TIMEOUT — treating as death")
                    return obs, self.death_penalty, True, False, {
                        "phase": "game_over",
                        "terminate_reason": "transition_timeout",
                        "score": curr_score,
                        "score_addr": self._score_addr,
                        "episode_reward": self._episode_reward,
                        "episode_steps": self._step_count,
                    }
                return obs, 0.0, False, False, {
                    "phase": "transition",
                    "score": curr_score,
                    "score_addr": self._score_addr,
                }

            self._consecutive_transition = 0

            # Execute action on every sub-frame
            self._execute_action(action)

            # On intermediate frames, just let the game advance
            if not last_frame:
                continue

        # Final frame: compute reward and return observation
        obs = self._preprocess(raw)
        reward, reward_info = self._compute_reward(action)
        self._step_count += 1
        self._episode_reward += reward
        self._log(1, f"[step {self._step_count}] action={self._format_action(action)} reward={reward:.4f} total={self._episode_reward:.4f}")
        self._log(
            2,
            (
                f"[step {self._step_count}] score={reward_info['score']} "
                f"score_delta={reward_info['score_delta']} score_reward={reward_info['score_reward']:.4f} "
                f"progress_reward={reward_info['progress_reward']:.4f} "
                f"time_penalty={reward_info['time_penalty']:.4f}"
            ),
        )
        info = {
            "phase": "in_game",
            "score_addr": self._score_addr,
            "episode_reward": self._episode_reward,
            "episode_steps": self._step_count,
            **reward_info,
        }
        return obs, reward, False, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._episode_count += 1
        self._log(
            1,
            f"[reset #{self._episode_count}] Loading save state... "
            f"last_end_reason={self._last_end_reason} prev_steps={self._step_count}",
        )
        self._maybe_enable_fast_forward()
        if self.reset_via_network:
            self._send_retroarch_cmd("LOAD_STATE")
        else:
            self._tap_key(self.load_state_key)
        time.sleep(0.5)
        self._prev_score = self._read_score()
        self._step_count = 0
        self._consecutive_transition = 0
        self._episode_reward = 0.0
        self._seen_alive = False

        # Wait until we see in-game screen (save state loads into gameplay)
        if self.in_game_checks:
            deadline = time.monotonic() + self.reset_wait_for_in_game_s
            while time.monotonic() < deadline:
                raw = self._grab_raw()
                if self._is_in_game(raw):
                    break
                time.sleep(self.reset_check_interval_s)
            else:
                self._log(1, "[reset] Timeout waiting for in-game; returning current frame.")

        raw = self._grab_raw()
        obs = self._preprocess(raw)
        return obs, {}

    def close(self):
        if self.sct is not None:
            self.sct.close()
        self._ram_sock.close()
        self._status_sock.close()
        self._cmd_sock.close()
