"""
Metal Slug gymnasium environment.

Action space: MultiDiscrete([5, 3, 3]) — movement, attack, and modifier
can be combined simultaneously (e.g. right+shoot+jump).

Game over detection: RAM-based via game_mode at 003868DA.
game_mode=1 means alive, game_mode!=1 means dead/menu.
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

pyautogui.PAUSE = 0  # eliminate hidden 100ms sleep after every keyDown/keyUp

from src.env.ram_decode import decode_bcd_score
from src.env.rewards import compute_reward


class XlibKeyboard:
    """Fast keyboard input via python-xlib XTest extension.

    Avoids pyautogui overhead (~0.1ms PAUSE + keycode lookup per call).
    Falls back to pyautogui if Xlib is unavailable (e.g. Wayland, no X11).
    """

    # Keys used by the game environment
    _KEY_NAMES = [
        "left", "right", "up", "down",
        "z", "x", "a", "s", "w",
        "f2", "f4", "tab", "Shift_R", "Return",
    ]

    def __init__(self):
        self._use_xlib = False
        self._display = None
        self._keycodes: dict[str, int] = {}
        try:
            from Xlib import X, display as xdisplay
            from Xlib.ext import xtest
            self._X = X
            self._xtest = xtest
            d = xdisplay.Display()
            # Pre-cache keycodes for all game keys
            for name in self._KEY_NAMES:
                keysym = self._name_to_keysym(name)
                if keysym is None:
                    continue
                kc = d.keysym_to_keycode(keysym)
                if kc:
                    self._keycodes[name.lower()] = kc
            self._display = d
            self._use_xlib = True
        except Exception:
            pass

    @staticmethod
    def _name_to_keysym(name: str):
        """Convert a key name to an X11 keysym."""
        from Xlib import XK
        # Map pyautogui-style names to Xlib keysym names
        mapping = {
            "left": "Left", "right": "Right", "up": "Up", "down": "Down",
            "tab": "Tab", "return": "Return", "shift_r": "Shift_R",
            "f2": "F2", "f4": "F4",
        }
        xlib_name = mapping.get(name.lower(), name)
        # Try XK_<name> lookup
        ks = XK.string_to_keysym(xlib_name)
        if ks == 0:
            # Single character keys (z, x, a, s, w)
            ks = XK.string_to_keysym(xlib_name.lower())
        return ks if ks != 0 else None

    def key_down(self, key: str):
        if self._use_xlib:
            kc = self._keycodes.get(key.lower())
            if kc:
                self._xtest.fake_input(self._display, self._X.KeyPress, kc)
                return
        pyautogui.keyDown(key)

    def key_up(self, key: str):
        if self._use_xlib:
            kc = self._keycodes.get(key.lower())
            if kc:
                self._xtest.fake_input(self._display, self._X.KeyRelease, kc)
                return
        pyautogui.keyUp(key)

    def flush(self):
        """Flush pending X events (non-blocking)."""
        if self._use_xlib and self._display:
            self._display.flush()


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
    REWARD_KEYS = (
        "score_reward", "progress_right", "progress_left", "time_penalty",
        "x_progress", "hp_loss", "game_over",
        "grenade_pickup", "grenade_waste", "ammo_pickup", "ammo_waste",
        "score_stall", "jump_bonus",
    )

    def __init__(
        self,
        region: CaptureRegion | None = None,
        action_hold_s: float = 0.005,
        frame_skip: int = 6,
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
        display: str | None = None,
        ram_read_retries: int = 3,
        player_x_addr: str | None = None,
        lives_addr: str = "003868D1",
        progress_scale: float = 0.01,
        score_scale: float = 0.002,
        score_clip: float = 2.0,
        time_penalty: float = -0.0005,
        hp_loss_penalties: dict[int, float] | None = None,
        game_over_penalty: float = -2.0,
        grenade_pickup_reward: float = 0.001,
        ammo_pickup_reward: float = 0.002,
        grenade_waste_penalty: float = -0.0001,
        ammo_waste_penalty: float = -0.00005,
        score_stall_threshold: int = 60,
        score_stall_penalty: float = -0.002,
        jump_bonus: float = 0.0,
        jump_bonus_stuck: float = 0.02,
        stuck_threshold_steps: int = 10,
        progress_scale_x: float = 0.01,
        max_episode_steps: int = 3000,
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
        # Confirmed RAM addresses for gameplay state
        self._lives_addr = "003D3B07"
        self._arms_addr = "003D3F48"
        self._bomb_addr = "003D3F45"
        self._time_addr = "003FB939"
        # Combined read: 1091 bytes from lives_addr covers lives(+0), bombs(+1086), arms(+1089..+1090)
        self._lives_block_addr = "003D3B07"
        self._lives_block_size = 1091
        self._use_individual_reads = False  # fallback if block read fails
        self.progress_scale = progress_scale
        self.score_scale = score_scale
        self.score_clip = score_clip
        self.time_penalty = time_penalty
        # HP loss penalties: keyed by HP value BEFORE the drop
        # e.g. {2: -1.0} means -1.0 when HP drops from 2 to 1
        self._hp_loss_penalties = hp_loss_penalties if hp_loss_penalties is not None else {2: -2.0, 1: -2.0}
        self._game_over_penalty = game_over_penalty
        # Resource management rewards
        self._grenade_pickup_reward = grenade_pickup_reward
        self._ammo_pickup_reward = ammo_pickup_reward
        self._grenade_waste_penalty = grenade_waste_penalty
        self._ammo_waste_penalty = ammo_waste_penalty
        self._score_stall_threshold = max(1, int(score_stall_threshold))
        self._score_stall_penalty = score_stall_penalty
        self._jump_bonus = jump_bonus
        self._jump_bonus_stuck = jump_bonus_stuck
        self._stuck_threshold_steps = max(1, int(stuck_threshold_steps))
        self._progress_scale_x = progress_scale_x
        self._max_episode_steps = max(0, int(max_episode_steps))
        self._score_stall_steps = 0
        self._stuck_steps = 0  # consecutive steps pressing right with no X increase
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
        self._reward_breakdown = {k: 0.0 for k in self.REWARD_KEYS}
        self._max_player_x = 0
        self._seen_alive = False  # track if game_mode==1 seen since last reset
        self._prev_lives: int | None = None
        self._prev_bombs: int | None = None
        self._prev_arms: int | None = None
        self._deaths_this_episode: int = 0
        self._score_addr = "003869BC"
        udp_timeout = float(os.environ.get("UDP_TIMEOUT", "0.01"))
        self._ram_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._ram_sock.settimeout(udp_timeout)
        # Dedicated socket for status block reads (death detection) to avoid
        # cross-contamination with other RAM reads sharing _ram_sock.
        self._status_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._status_sock.settimeout(udp_timeout)
        self._cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._cmd_sock.settimeout(udp_timeout)
        self.action_space = gym.spaces.MultiDiscrete([5, 3, 3])
        self.observation_space = gym.spaces.Box(0, 255, shape=(84, 84, 1), dtype=np.uint8)
        # Set DISPLAY before creating mss/Xlib handles so each SubprocVecEnv
        # worker captures from its own Xvfb instance.
        if display is not None:
            os.environ['DISPLAY'] = display
        self.sct = mss.mss()
        pyautogui.FAILSAFE = True
        self._kb = XlibKeyboard()

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
        self._kb.key_down(key)
        self._kb.flush()
        time.sleep(self.action_hold_s)
        self._kb.key_up(key)
        self._kb.flush()

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

    def _read_status_and_score(self) -> tuple[int | None, int | None, int]:
        """Batch-read game_state, game_mode, and score in a single UDP request.

        Reads 241 bytes from 003868D0, covering:
          - 003868D0 (offset 0): game_state
          - 003868DA (offset 10): game_mode
          - 003869BC (offset 236..240): score (4 BCD bytes + 1 upper byte)
        Returns (game_state, game_mode, score).
        """
        for _ in range(self.ram_read_retries):
            try:
                cmd = "READ_CORE_RAM 003868D0 241\n"
                self._status_sock.sendto(cmd.encode(), (self.retroarch_host, self.retroarch_cmd_port))
                data, _ = self._status_sock.recvfrom(2048)
                parts = data.decode().strip().split()
                values = [int(b, 16) for b in parts[2:]]
                if len(values) == 241:
                    game_state = values[0]
                    game_mode = values[10]
                    score_bytes = values[236:241]
                    decoded = decode_bcd_score(score_bytes)
                    score = decoded if decoded is not None else self._prev_score
                    return game_state, game_mode, score
            except socket.timeout:
                continue
        return None, None, self._prev_score

    def _is_ram_death(self) -> bool:
        """Detect death via game_mode at 003868DA.

        game_mode=1 means alive/playing. Any other value means the player
        is dead or at a non-gameplay screen:
          0 = title/continue screen
          2 = death animation
          3 = transition

        After reset, game_mode may not settle to 1 immediately.
        We wait until we've seen game_mode==1 at least once before triggering
        death detection, to avoid false positives.
        """
        gs, game_mode = self._read_status_block()
        self._log(2, f"[death_check] gs={gs} game_mode={game_mode} seen_alive={self._seen_alive}")
        if game_mode is None:
            return False
        if not self._seen_alive:
            if game_mode == 1:
                self._seen_alive = True
            return False
        return game_mode != 1

    def _check_death_and_score(self) -> tuple[bool, int]:
        """Batched death check + score read in a single UDP request.

        Returns (is_dead, score). Used on the last frame of frame_skip
        to avoid two separate UDP round-trips.
        """
        gs, game_mode, score = self._read_status_and_score()
        self._log(2, f"[death_check] gs={gs} game_mode={game_mode} seen_alive={self._seen_alive}")
        if game_mode is None:
            return False, score
        if not self._seen_alive:
            if game_mode == 1:
                self._seen_alive = True
            return False, score
        return game_mode != 1, score

    def _read_ram(self, address, num_bytes=4):
        # Buffer must fit header + "XX " per byte; 4*num_bytes is safe with margin
        recv_buf = max(1024, num_bytes * 4 + 64)
        for _ in range(self.ram_read_retries):
            try:
                cmd = f"READ_CORE_RAM {address} {num_bytes}\n"
                self._ram_sock.sendto(cmd.encode(), (self.retroarch_host, self.retroarch_cmd_port))
                data, _ = self._ram_sock.recvfrom(recv_buf)
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
        b = self._read_ram(self._score_addr, 5)
        decoded = decode_bcd_score(b)
        if decoded is None:
            return self._prev_score
        return decoded

    def _read_lives_block(self) -> tuple[int | None, int | None, int | None]:
        """Read lives, bombs, and arms in a single 1091-byte RAM read.

        Layout from 003D3B07:
          offset 0:    lives (1 byte)
          offset 1086: bombs (1 byte, addr 003D3F45)
          offset 1089..1090: arms (2 bytes LE, addr 003D3F48)
        Returns (lives, bombs, arms) or (None, None, None) on failure.
        """
        if self._use_individual_reads:
            return self._read_lives_individual()
        values = self._read_ram(self._lives_block_addr, self._lives_block_size)
        if not values or len(values) != self._lives_block_size:
            if values and len(values) != self._lives_block_size:
                # Block read returned wrong size — switch to individual reads
                self._log(1, "[ram] Lives block read returned %d bytes (expected %d), switching to individual reads"
                          % (len(values), self._lives_block_size))
                self._use_individual_reads = True
            return self._read_lives_individual()
        lives = values[0]
        bombs = values[1086]
        arms = values[1089] | (values[1090] << 8)
        return lives, bombs, arms

    def _read_lives_individual(self) -> tuple[int | None, int | None, int | None]:
        """Fallback: read lives, bombs, arms as 3 separate small reads."""
        lives_raw = self._read_ram(self._lives_addr, 1)
        lives = lives_raw[0] if lives_raw else None
        bomb_raw = self._read_ram(self._bomb_addr, 1)
        bombs = bomb_raw[0] if bomb_raw else None
        arms_raw = self._read_ram(self._arms_addr, 2)
        arms = (arms_raw[0] | (arms_raw[1] << 8)) if arms_raw and len(arms_raw) == 2 else None
        return lives, bombs, arms

    def _read_time(self) -> int | None:
        """Read in-game timer (BCD) from 003FB939. Returns raw byte or None."""
        values = self._read_ram(self._time_addr, 1)
        if values and len(values) == 1:
            return values[0]
        return None

    def _read_extended_state(self) -> tuple[int | None, int | None, int | None, int | None]:
        """Read lives, bombs, and arms. Returns (lives, bombs, arms, time).

        game_time is only used in the info dict, not for rewards — skip the
        UDP round-trip and return None to save ~1ms per step.
        """
        lives, bombs, arms = self._read_lives_block()
        return lives, bombs, arms, None

    def _handle_death_event(self) -> float:
        """Handle a game_mode death event (game_mode != 1).

        game_mode != 1 means the player is dead or at a non-gameplay screen —
        apply game_over_penalty and always terminate the episode.

        Also retroactively applies any missed HP drop penalties. The HP RAM
        byte can skip straight from 2 to 0 (or the reads can be missed due to
        frame timing), so we walk _prev_lives down to 0 and apply each
        intermediate penalty that wasn't already triggered.

        Returns the total penalty (game_over + missed HP drops).
        """
        self._deaths_this_episode += 1
        penalty = self._game_over_penalty

        # Retroactively apply missed HP drops: walk prev_lives down to 0
        missed_hp = 0.0
        if self._prev_lives is not None and self._prev_lives > 0:
            hp = self._prev_lives
            while hp > 0:
                hp_pen = self._hp_loss_penalties.get(hp, 0.0)
                if hp_pen != 0.0:
                    missed_hp += hp_pen
                    self._track("hp_loss", hp_pen)
                hp -= 1
            self._prev_lives = 0

        total = penalty + missed_hp
        self._log(1, "[death] GAME OVER penalty=%.1f missed_hp=%.1f total=%.1f deaths=%d"
                  % (penalty, missed_hp, total, self._deaths_this_episode))
        return total

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

    def _track(self, key: str, value: float):
        self._reward_breakdown[key] += value

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

    def _compute_reward_with_score(self, action, curr_score: int):
        """Compute reward using an already-read score (avoids extra UDP call)."""
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

    def _release_all_keys(self):
        """Release all game-relevant keys to prevent sticky keys."""
        for k in ["left", "right", "up", "down",
                   self.shoot_key, self.grenade_key,
                   self.jump_key, self.metal_slug_attack_key]:
            self._kb.key_up(k)
        self._kb.flush()

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
            self._kb.key_down(k)
        self._kb.flush()
        time.sleep(self.action_hold_s)
        for k in keys:
            self._kb.key_up(k)
        self._kb.flush()

    def step(self, action):
        # SB3 passes MultiDiscrete as numpy array; ensure flat int array.
        action = np.asarray(action).flatten().astype(int)

        # Safety: if we've never seen game_mode==1 and taken 50+ steps,
        # the game is stuck at menu/loading. Force terminate so reset() retries.
        if not self._seen_alive and self._step_count >= 50:
            self._log(1, f"[step {self._step_count}] Never saw game_mode==1, force terminating (stuck at menu)")
            raw = self._grab_raw()
            obs = self._preprocess(raw)
            self._last_end_reason = "never_alive"
            penalty = self._game_over_penalty
            self._episode_reward += penalty
            self._track("game_over", penalty)
            return obs, penalty, True, False, {
                "phase": "game_over",
                "terminate_reason": "never_alive",
                "score": 0,
                "episode_reward": self._episode_reward,
                "episode_steps": self._step_count,
                "deaths_this_episode": 0,
                "lives": None, "bombs": None, "arms": None, "game_time": None,
                "reward_breakdown": dict(self._reward_breakdown),
                "max_player_x": self._max_player_x,
            }

        # Frame skip: repeat the action multiple times, checking for death
        # via RAM each iteration but only capturing the screen on the last
        # frame (or on early termination).  Skipping intermediate grabs
        # eliminates ~75% of mss.grab() overhead.
        #
        # On the last frame we use _check_death_and_score() which batches
        # game_state + game_mode + score into a single UDP request, saving
        # one round-trip per step.
        for skip_i in range(self.frame_skip):
            last_frame = (skip_i == self.frame_skip - 1)

            if not last_frame:
                # Intermediate frame: just execute the action, no UDP reads.
                # Death state (game_mode!=1) persists for many frames during
                # the death animation and will be caught on the last frame.
                self._execute_action(action)
                continue

            # Last frame: batched death+score read (1 UDP request instead of 2)
            is_dead, curr_score = self._check_death_and_score()
            if is_dead:
                penalty = self._handle_death_event()
                raw = self._grab_raw()
                obs = self._preprocess(raw)
                self._episode_reward += penalty
                self._track("game_over", self._game_over_penalty)
                self._last_end_reason = "ram_death"
                self._log(1, f"[step {self._step_count}] GAME OVER (game_mode!=1), penalty={penalty}")
                return obs, penalty, True, False, {
                    "phase": "game_over",
                    "terminate_reason": "ram_death",
                    "score": curr_score,
                    "score_addr": self._score_addr,
                    "episode_reward": self._episode_reward,
                    "episode_steps": self._step_count,
                    "deaths_this_episode": self._deaths_this_episode,
                    "lives": None, "bombs": None, "arms": None, "game_time": None,
                    "reward_breakdown": dict(self._reward_breakdown),
                    "max_player_x": self._max_player_x,
                }

            self._execute_action(action)

            # Last frame: grab screen for observation and in-game checks
            raw = self._grab_raw()

            if self.enforce_in_game_checks and self.in_game_checks and not self._is_in_game(raw):
                self._consecutive_transition += 1
                obs = self._preprocess(raw)
                self._log(1, f"[step {self._step_count}] TRANSITION [{self._consecutive_transition}/{self.max_transition_steps}]")
                if self.max_transition_steps > 0 and self._consecutive_transition >= self.max_transition_steps:
                    self._last_end_reason = "transition_timeout"
                    penalty = self._game_over_penalty
                    self._episode_reward += penalty
                    self._track("game_over", penalty)
                    self._log(1, f"[step {self._step_count}] TRANSITION TIMEOUT — treating as death")
                    return obs, penalty, True, False, {
                        "phase": "game_over",
                        "terminate_reason": "transition_timeout",
                        "score": curr_score,
                        "score_addr": self._score_addr,
                        "episode_reward": self._episode_reward,
                        "episode_steps": self._step_count,
                        "deaths_this_episode": self._deaths_this_episode,
                        "lives": None, "bombs": None, "arms": None, "game_time": None,
                        "reward_breakdown": dict(self._reward_breakdown),
                        "max_player_x": self._max_player_x,
                    }
                return obs, 0.0, False, False, {
                    "phase": "transition",
                    "score": curr_score,
                    "score_addr": self._score_addr,
                    "deaths_this_episode": self._deaths_this_episode,
                    "lives": None, "bombs": None, "arms": None, "game_time": None,
                }

            self._consecutive_transition = 0

        # Final frame: compute reward using already-read score and return observation
        obs = self._preprocess(raw)
        reward, reward_info = self._compute_reward_with_score(action, curr_score)
        self._step_count += 1
        self._episode_reward += reward

        # Track base reward components
        self._track("score_reward", reward_info["score_reward"])
        pr = reward_info["progress_reward"]
        if pr > 0:
            self._track("progress_right", pr)
        elif pr < 0:
            self._track("progress_left", pr)
        self._track("time_penalty", reward_info["time_penalty"])

        # Read extended state (lives/HP, bombs, arms, time) for info dict + HP tracking
        lives, bombs, arms, game_time = self._read_extended_state()

        # Player X tracking: stuck detection and optional X-based progress reward
        # Default addr 3FB84E is per-screen scroll X (resets at screen transitions)
        curr_x = self._read_player_x()
        is_stuck = False
        if self.player_x_addr:
            if curr_x is not None and self._prev_x is not None:
                x_delta = curr_x - self._prev_x
                moving_right = (action[0] == 2)
                # Large negative delta = screen transition (scroll reset), not stuck
                if x_delta < -100:
                    self._stuck_steps = 0
                elif moving_right and x_delta <= 0:
                    self._stuck_steps += 1
                else:
                    self._stuck_steps = 0
                is_stuck = self._stuck_steps >= self._stuck_threshold_steps
                # Optional X-based progress reward (disabled by default for scroll addr)
                if x_delta > 0 and self._progress_scale_x > 0:
                    x_progress = min(x_delta * self._progress_scale_x, 0.5)
                    reward += x_progress
                    self._episode_reward += x_progress
                    self._track("x_progress", x_progress)
            if curr_x is not None:
                self._prev_x = curr_x
                if curr_x < 10000 and curr_x > self._max_player_x:
                    self._max_player_x = curr_x
        else:
            # Fallback: treat score stall as stuck proxy
            is_stuck = self._score_stall_steps >= self._score_stall_threshold

        # HP loss detection: apply incremental penalties when HP drops during gameplay.
        # Enforce monotonic decrease — HP can only go 2→1→0, never back up.
        # The RAM byte at 003D3B07 oscillates during gameplay, so we ignore
        # any read that is higher than _prev_lives (treat as flicker).
        # Walk through all intermediate HP values to catch jumps (e.g. 2→0
        # should apply penalties for both 2→1 and 1→0).
        if lives is not None and self._prev_lives is not None:
            if lives < self._prev_lives:
                hp = self._prev_lives
                num_penalties = 0
                step_hp_total = 0.0
                while hp > lives:
                    hp_penalty = self._hp_loss_penalties.get(hp, 0.0)
                    if hp_penalty != 0.0:
                        num_penalties += 1
                        step_hp_total += hp_penalty
                        self._log(1, f"[step {self._step_count}] HP DROP {hp}->{hp - 1} penalty={hp_penalty}")
                        reward += hp_penalty
                        self._episode_reward += hp_penalty
                        self._track("hp_loss", hp_penalty)
                    hp -= 1
                # Only update prev_lives on decrease — ignore upward flickers
                self._prev_lives = lives

        # Resource management: reward pickups, penalize wasteful use
        # Guard against RAM misreads with max delta thresholds
        score_delta = reward_info["score_delta"]
        if bombs is not None and self._prev_bombs is not None:
            bomb_diff = bombs - self._prev_bombs
            if abs(bomb_diff) <= 10:  # ignore junk reads
                if bomb_diff > 0:
                    bonus = self._grenade_pickup_reward * bomb_diff
                    reward += bonus
                    self._episode_reward += bonus
                    self._track("grenade_pickup", bonus)
                    self._log(1, f"[step {self._step_count}] GRENADE PICKUP +{bomb_diff} reward={bonus}")
                elif bomb_diff < 0 and score_delta == 0:
                    penalty = self._grenade_waste_penalty * (-bomb_diff)
                    reward += penalty
                    self._episode_reward += penalty
                    self._track("grenade_waste", penalty)
                    self._log(1, f"[step {self._step_count}] GRENADE WASTED {bomb_diff} penalty={penalty}")
        if bombs is not None:
            self._prev_bombs = bombs

        if arms is not None and self._prev_arms is not None and arms != 65535 and self._prev_arms != 65535:
            arms_diff = arms - self._prev_arms
            if abs(arms_diff) <= 100:  # ignore junk reads (arms is 16-bit LE)
                if arms_diff > 0:
                    bonus = self._ammo_pickup_reward * arms_diff
                    reward += bonus
                    self._episode_reward += bonus
                    self._track("ammo_pickup", bonus)
                    self._log(1, f"[step {self._step_count}] AMMO PICKUP +{arms_diff} reward={bonus}")
                elif arms_diff < 0 and score_delta == 0:
                    penalty = self._ammo_waste_penalty * (-arms_diff)
                    reward += penalty
                    self._episode_reward += penalty
                    self._track("ammo_waste", penalty)
                    self._log(1, f"[step {self._step_count}] AMMO WASTED {arms_diff} penalty={penalty}")
        if arms is not None:
            self._prev_arms = arms

        # Score stall penalty: punish when score stays flat for too long (stuck/idle)
        if score_delta == 0:
            self._score_stall_steps += 1
        else:
            self._score_stall_steps = 0
        if self._score_stall_steps >= self._score_stall_threshold:
            reward += self._score_stall_penalty
            self._episode_reward += self._score_stall_penalty
            self._track("score_stall", self._score_stall_penalty)
            if self._score_stall_steps == self._score_stall_threshold:
                self._log(1, f"[step {self._step_count}] [SCORE STALL] no score increase for {self._score_stall_steps} steps, applying penalty={self._score_stall_penalty}")

        # Jump bonus: larger when stuck to encourage jumping over obstacles
        if action[2] == 1:  # modifier=jump
            jb = self._jump_bonus_stuck if is_stuck else self._jump_bonus
            reward += jb
            self._episode_reward += jb
            self._track("jump_bonus", jb)
            if is_stuck:
                self._log(1, f"[step {self._step_count}] STUCK JUMP bonus={jb} stuck_steps={self._stuck_steps}")

        self._log(1, f"[step {self._step_count}] action={self._format_action(action)} reward={reward:.4f} total={self._episode_reward:.4f}")
        self._log(
            2,
            (
                f"[step {self._step_count}] score={reward_info['score']} "
                f"score_delta={reward_info['score_delta']} score_reward={reward_info['score_reward']:.4f} "
                f"progress_reward={reward_info['progress_reward']:.4f} "
                f"time_penalty={reward_info['time_penalty']:.4f} "
                f"hp={lives} bombs={bombs} arms={arms} time={game_time}"
            ),
        )
        # Max episode steps truncation: safety valve to prevent infinite episodes.
        # Uses truncated=True (not terminated) so PPO bootstraps the value estimate.
        truncated = self._max_episode_steps > 0 and self._step_count >= self._max_episode_steps
        if truncated:
            self._log(1, f"[step {self._step_count}] MAX EPISODE STEPS reached ({self._max_episode_steps}), truncating")
            self._last_end_reason = "max_steps"

        info = {
            "phase": "in_game",
            "score_addr": self._score_addr,
            "episode_reward": self._episode_reward,
            "episode_steps": self._step_count,
            "deaths_this_episode": self._deaths_this_episode,
            "lives": lives,
            "bombs": bombs,
            "arms": arms,
            "game_time": game_time,
            "is_stuck": is_stuck,
            "player_x": curr_x,
            **reward_info,
        }
        if truncated:
            info["terminate_reason"] = "max_steps"
            info["reward_breakdown"] = dict(self._reward_breakdown)
            info["max_player_x"] = self._max_player_x
        return obs, reward, False, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._release_all_keys()
        self._episode_count += 1
        self._log(
            1,
            f"[reset #{self._episode_count}] Loading save state... "
            f"last_end_reason={self._last_end_reason} prev_steps={self._step_count}",
        )
        self._maybe_enable_fast_forward()

        # Load save state and verify we reach gameplay (game_mode==1).
        # Retry LOAD_STATE up to 3 times if game_mode doesn't settle to 1.
        max_load_attempts = 3
        for load_attempt in range(max_load_attempts):
            if self.reset_via_network:
                self._send_retroarch_cmd("LOAD_STATE")
            else:
                self._tap_key(self.load_state_key)
            time.sleep(0.5)

            # Poll game_mode for up to 5 seconds to confirm we're in gameplay
            deadline = time.monotonic() + 5.0
            got_alive = False
            while time.monotonic() < deadline:
                _, game_mode = self._read_status_block()
                if game_mode == 1:
                    got_alive = True
                    break
                time.sleep(0.1)

            if got_alive:
                self._log(1, f"[reset] LOAD_STATE succeeded on attempt {load_attempt + 1}")
                break
            else:
                self._log(1, f"[reset] LOAD_STATE attempt {load_attempt + 1}/{max_load_attempts} "
                          f"failed (game_mode={game_mode}), retrying...")
        else:
            self._log(1, "[reset] WARNING: game_mode never reached 1 after all LOAD_STATE attempts")

        self._prev_score = self._read_score()
        self._step_count = 0
        self._consecutive_transition = 0
        self._episode_reward = 0.0
        self._reward_breakdown = {k: 0.0 for k in self.REWARD_KEYS}
        self._max_player_x = 0
        self._seen_alive = got_alive
        self._deaths_this_episode = 0
        self._score_stall_steps = 0
        self._stuck_steps = 0
        self._use_individual_reads = False  # reset fallback flag each episode
        # Read initial player X position for stuck detection
        self._prev_x = self._read_player_x()
        # Read initial HP and resources (003D3B07 starts at 2 = full health)
        # Always init _prev_lives=2: save state is always created at full HP,
        # and the RAM byte may not be populated immediately after LOAD_STATE.
        # A stale read of 0 would disable HP penalty tracking for the episode.
        lives, bombs, arms = self._read_lives_block()
        self._prev_lives = 2
        self._prev_bombs = bombs
        self._prev_arms = arms

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
