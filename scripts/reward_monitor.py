#!/usr/bin/env python3
"""Standalone RAM reward monitor for Metal Slug 6.

Polls RAM via RetroArch UDP at configurable Hz. Detects death when
lives stably reaches 0 (all 3 lives used). Resets via LOAD_STATE.

RAM values are stabilized: a value must appear for 3 consecutive
reads before being accepted, filtering out transient noise.
"""

import argparse
import socket
import time
import sys

sys.path.insert(0, ".")
from src.env.ram_decode import decode_bcd_score

STABLE_THRESHOLD = 3  # consecutive identical reads to accept a value


class StableValue:
    """Track a RAM value, only updating when stable across multiple reads."""

    def __init__(self):
        self.confirmed = None  # last accepted stable value
        self._pending = None
        self._pending_count = 0

    def update(self, raw):
        """Feed a new read. Returns True if confirmed value changed."""
        if raw is None:
            return False
        if raw == self._pending:
            self._pending_count += 1
        else:
            self._pending = raw
            self._pending_count = 1
        if self._pending_count >= STABLE_THRESHOLD and self._pending != self.confirmed:
            old = self.confirmed
            self.confirmed = self._pending
            return old is not None  # suppress first-ever confirmation
        return False

    def reset(self):
        self.confirmed = None
        self._pending = None
        self._pending_count = 0


def read_ram(sock, host, port, address, num_bytes, retries=3):
    for _ in range(retries):
        try:
            cmd = f"READ_CORE_RAM {address} {num_bytes}\n"
            sock.sendto(cmd.encode(), (host, port))
            data, _ = sock.recvfrom(1024)
            parts = data.decode().strip().split()
            values = [int(b, 16) for b in parts[2:]]
            if len(values) == num_bytes:
                return values
        except socket.timeout:
            continue
    return []


def send_cmd(sock, host, port, command):
    cmd = command.strip().upper() + "\n"
    sock.sendto(cmd.encode(), (host, port))


def main():
    parser = argparse.ArgumentParser(description="RAM-based reward monitor for Metal Slug 6")
    parser.add_argument("--host", default="127.0.0.1", help="RetroArch host")
    parser.add_argument("--port", type=int, default=55355, help="RetroArch UDP command port")
    parser.add_argument("--poll-hz", type=float, default=30, help="Polling frequency in Hz")
    parser.add_argument("--reset-delay", type=float, default=3.0,
                        help="Seconds to wait after game over before LOAD_STATE reset")
    parser.add_argument("--max-episodes", type=int, default=0,
                        help="Stop after N episodes (0 = unlimited)")
    parser.add_argument("--game-state-addr", default="003868D0", help="RAM address for game_state byte")
    parser.add_argument("--lives-addr", default="003D3B07", help="RAM address for lives byte (actual counter)")
    parser.add_argument("--bomb-addr", default="003D3F45", help="RAM address for bombs byte")
    parser.add_argument("--arms-addr", default="003D3F48", help="RAM address for arms/ammo (2 bytes LE)")
    parser.add_argument("--time-addr", default="003FB939", help="RAM address for in-game timer (BCD)")
    parser.add_argument("--score-addr", default="003869BC", help="RAM address for 4-byte BCD score")
    args = parser.parse_args()

    poll_interval = 1.0 / args.poll_hz
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(0.1)

    print(f"Reward monitor started: {args.host}:{args.port} @ {args.poll_hz}Hz")
    print(f"  lives_addr={args.lives_addr}  game_state_addr={args.game_state_addr}")
    print(f"  bomb_addr={args.bomb_addr}  arms_addr={args.arms_addr}  time_addr={args.time_addr}")
    print(f"  score_addr={args.score_addr}")
    print(f"  reset_delay={args.reset_delay}s")
    print(f"  stable_threshold={STABLE_THRESHOLD}")
    print()

    episode = 0
    ep_start = time.monotonic()
    lives_seen_positive = False

    stable_lives = StableValue()
    stable_gs = StableValue()
    stable_bombs = StableValue()
    stable_arms = StableValue()
    stable_time = StableValue()

    # Read initial score
    score_raw = read_ram(sock, args.host, args.port, args.score_addr, 4)
    prev_score = decode_bcd_score(score_raw) or 0

    try:
        while True:
            t0 = time.monotonic()
            elapsed = t0 - ep_start

            # Read RAM
            lives_raw = read_ram(sock, args.host, args.port, args.lives_addr, 1)
            gs_raw = read_ram(sock, args.host, args.port, args.game_state_addr, 1)
            bomb_raw = read_ram(sock, args.host, args.port, args.bomb_addr, 1)
            arms_raw = read_ram(sock, args.host, args.port, args.arms_addr, 2)
            time_raw = read_ram(sock, args.host, args.port, args.time_addr, 1)
            score_raw = read_ram(sock, args.host, args.port, args.score_addr, 4)

            lives_val = lives_raw[0] if lives_raw else None
            gs_val = gs_raw[0] if gs_raw else None
            bomb_val = bomb_raw[0] if bomb_raw else None
            arms_val = (arms_raw[0] | (arms_raw[1] << 8)) if arms_raw and len(arms_raw) == 2 else None
            time_val = time_raw[0] if time_raw else None
            score = decode_bcd_score(score_raw)

            # Update stable trackers and log confirmed changes
            lives_changed = stable_lives.update(lives_val)
            gs_changed = stable_gs.update(gs_val)
            bombs_changed = stable_bombs.update(bomb_val)
            arms_changed = stable_arms.update(arms_val)
            time_changed = stable_time.update(time_val)

            if lives_changed:
                gs_str = f"gs=0x{stable_gs.confirmed:02X}" if stable_gs.confirmed is not None else ""
                print(f"  [{elapsed:7.1f}s] lives -> {stable_lives.confirmed}  {gs_str}")

            if gs_changed:
                gs_hex = f"0x{stable_gs.confirmed:02X}" if stable_gs.confirmed is not None else "??"
                print(f"  [{elapsed:7.1f}s] game_state -> {gs_hex}")

            if bombs_changed:
                print(f"  [{elapsed:7.1f}s] bombs -> {stable_bombs.confirmed}")

            if arms_changed:
                print(f"  [{elapsed:7.1f}s] arms -> {stable_arms.confirmed}")

            if time_changed:
                print(f"  [{elapsed:7.1f}s] time -> {stable_time.confirmed}")

            # Score changes (no stabilization needed — BCD decode filters garbage)
            if score is not None:
                score_delta = score - prev_score
                if 0 < score_delta < 50000:
                    print(f"  [{elapsed:7.1f}s] SCORE +{score_delta} (total: {score:,})")
                prev_score = score

            # Track whether lives has been seen > 0 since last reset
            if stable_lives.confirmed is not None and stable_lives.confirmed > 0:
                lives_seen_positive = True

            # Death: stable lives reaches 0 (all 3 lives used up)
            # Only trigger after lives was confirmed > 0 to avoid false triggers
            # from noisy reads right after save state load.
            if lives_seen_positive and stable_lives.confirmed is not None and stable_lives.confirmed == 0:
                episode += 1
                ep_elapsed = time.monotonic() - ep_start
                score_str = f"{prev_score:,}"

                gs_hex = f"0x{stable_gs.confirmed:02X}" if stable_gs.confirmed is not None else "??"
                print(f"\n{'='*60}")
                print(f"DEATH — Episode {episode}")
                print(f"  score={score_str}  time={ep_elapsed:.1f}s  gs={gs_hex}")
                print(f"{'='*60}\n")

                if args.max_episodes > 0 and episode >= args.max_episodes:
                    print(f"Reached max_episodes={args.max_episodes}. Stopping.")
                    break

                print(f"Waiting {args.reset_delay}s before reset...")
                time.sleep(args.reset_delay)

                send_cmd(sock, args.host, args.port, "LOAD_STATE")
                print("LOAD_STATE sent. Waiting for game to resume...\n")
                time.sleep(1.0)

                # Reset episode tracking
                lives_seen_positive = False
                stable_lives.reset()
                stable_gs.reset()
                stable_bombs.reset()
                stable_arms.reset()
                stable_time.reset()
                ep_start = time.monotonic()
                score_raw = read_ram(sock, args.host, args.port, args.score_addr, 4)
                prev_score = decode_bcd_score(score_raw) or 0
                continue

            # Throttle to target Hz
            dt = time.monotonic() - t0
            sleep_time = poll_interval - dt
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
