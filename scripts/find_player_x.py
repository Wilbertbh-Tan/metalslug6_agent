#!/usr/bin/env python3
"""Find the player X RAM address by scanning for values that change with movement.

This script must run INSIDE the Docker container (where it can reach RetroArch's
UDP command port on 127.0.0.1:55355).

Algorithm:
  1. Load save state to get a consistent starting point
  2. Take RAM snapshot A (idle)
  3. Send "right" keypresses for a few seconds
  4. Take RAM snapshot B
  5. Filter for 16-bit LE values where B > A (moved right → value increased)
  6. Send more "right" keypresses
  7. Take RAM snapshot C
  8. Filter for addresses where C > B > A with consistent positive deltas
  9. Stand still, take snapshot D
  10. Filter for addresses where D == C (stopped moving → value stable)

Usage:
    python scripts/find_player_x.py [--region START END]

Default scan region: 003D0000 to 003E0000 (where other gameplay state lives).
Also scans 00380000-00390000 (where game_state/score live).
"""
import argparse
import os
import socket
import struct
import sys
import time

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


def read_ram_block(sock, host, port, start_addr, num_bytes, retries=3, timeout=1.0):
    """Read a block of RAM. Returns list of byte values or None."""
    old_timeout = sock.gettimeout()
    sock.settimeout(timeout)
    recv_buf = max(4096, num_bytes * 4 + 128)
    for _ in range(retries):
        try:
            cmd = "READ_CORE_RAM %06X %d\n" % (start_addr, num_bytes)
            sock.sendto(cmd.encode(), (host, port))
            data, _ = sock.recvfrom(recv_buf)
            parts = data.decode().strip().split()
            if parts[2] == "-1":
                return None  # address out of range
            values = [int(b, 16) for b in parts[2:]]
            if len(values) == num_bytes:
                sock.settimeout(old_timeout)
                return values
        except socket.timeout:
            continue
        except Exception as e:
            print("  Error reading %06X: %s" % (start_addr, e))
            continue
    sock.settimeout(old_timeout)
    return None


def snapshot_region(sock, host, port, start, end, chunk_size=256):
    """Read a RAM region in chunks, return dict of {addr: byte_value}."""
    data = {}
    addr = start
    while addr < end:
        size = min(chunk_size, end - addr)
        values = read_ram_block(sock, host, port, addr, size)
        if values is None:
            addr += size
            continue
        for i, v in enumerate(values):
            data[addr + i] = v
        addr += size
    return data


def extract_16bit_le(data, addr):
    """Extract a 16-bit little-endian value from snapshot data."""
    if addr in data and (addr + 1) in data:
        return data[addr] | (data[addr + 1] << 8)
    return None


def send_key(key, hold_s=0.05):
    """Send a keypress via pyautogui."""
    import pyautogui
    pyautogui.PAUSE = 0
    pyautogui.keyDown(key)
    time.sleep(hold_s)
    pyautogui.keyUp(key)


def hold_key(key, duration_s):
    """Hold a key for a duration."""
    import pyautogui
    pyautogui.PAUSE = 0
    pyautogui.keyDown(key)
    time.sleep(duration_s)
    pyautogui.keyUp(key)


def load_state(sock, host, port):
    """Send LOAD_STATE command."""
    cmd = "LOAD_STATE\n"
    sock.sendto(cmd.encode(), (host, port))
    time.sleep(1.0)


def main():
    parser = argparse.ArgumentParser(description="Find player X RAM address")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=55355)
    parser.add_argument("--regions", nargs="*", default=None,
                        help="Hex address pairs: START1 END1 START2 END2 ...")
    parser.add_argument("--chunk", type=int, default=256,
                        help="Bytes per UDP read (default: 256)")
    parser.add_argument("--move-duration", type=float, default=2.0,
                        help="Seconds to hold right between snapshots")
    args = parser.parse_args()

    # Default scan regions: where known gameplay addresses live
    if args.regions:
        hex_vals = [int(x, 16) for x in args.regions]
        regions = [(hex_vals[i], hex_vals[i+1]) for i in range(0, len(hex_vals), 2)]
    else:
        regions = [
            (0x00380000, 0x003A0000),  # game_state, score area
            (0x003D0000, 0x00400000),  # lives, bombs, arms, time area
        ]

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(0.5)

    total_bytes = sum(end - start for start, end in regions)
    print("Scanning %d bytes across %d region(s):" % (total_bytes, len(regions)))
    for start, end in regions:
        print("  %06X - %06X (%d bytes)" % (start, end, end - start))

    # Step 1: Load state for consistent starting point
    print("\n[1/7] Loading save state...")
    load_state(sock, args.host, args.port)
    time.sleep(0.5)

    # Step 2: Take snapshot A (idle)
    print("[2/7] Taking snapshot A (idle)...")
    snap_a = {}
    for start, end in regions:
        snap_a.update(snapshot_region(sock, args.host, args.port, start, end, args.chunk))
    print("  Read %d bytes" % len(snap_a))

    # Step 3: Move right
    print("[3/7] Moving right for %.1fs..." % args.move_duration)
    hold_key("right", args.move_duration)
    time.sleep(0.2)

    # Step 4: Take snapshot B
    print("[4/7] Taking snapshot B (after moving right)...")
    snap_b = {}
    for start, end in regions:
        snap_b.update(snapshot_region(sock, args.host, args.port, start, end, args.chunk))
    print("  Read %d bytes" % len(snap_b))

    # Step 5: Find 16-bit LE values where B > A (candidate X coordinates)
    candidates_ab = []
    all_addrs = sorted(set(snap_a.keys()) & set(snap_b.keys()))
    for addr in all_addrs:
        if (addr + 1) not in snap_a or (addr + 1) not in snap_b:
            continue
        val_a = snap_a[addr] | (snap_a[addr + 1] << 8)
        val_b = snap_b[addr] | (snap_b[addr + 1] << 8)
        delta = val_b - val_a
        # Filter: value increased by a reasonable amount (1-2000 pixels)
        if 1 <= delta <= 2000:
            candidates_ab.append((addr, val_a, val_b, delta))

    print("  Round 1: %d candidates where B > A (moved right → value increased)" % len(candidates_ab))
    if not candidates_ab:
        print("  No candidates found! The player X address might not be in the scanned regions.")
        print("  Try expanding the region with --regions.")
        sock.close()
        return

    # Step 6: Move right more
    print("[5/7] Moving right again for %.1fs..." % args.move_duration)
    hold_key("right", args.move_duration)
    time.sleep(0.2)

    # Step 7: Take snapshot C
    print("[6/7] Taking snapshot C (after moving right again)...")
    snap_c = {}
    for start, end in regions:
        snap_c.update(snapshot_region(sock, args.host, args.port, start, end, args.chunk))
    print("  Read %d bytes" % len(snap_c))

    # Filter: C > B > A with similar delta direction
    candidates_abc = []
    for addr, val_a, val_b, delta_ab in candidates_ab:
        if (addr + 1) not in snap_c:
            continue
        val_c = snap_c[addr] | (snap_c[addr + 1] << 8)
        delta_bc = val_c - val_b
        # C > B and delta is in reasonable range
        if 1 <= delta_bc <= 2000:
            candidates_abc.append((addr, val_a, val_b, val_c, delta_ab, delta_bc))

    print("  Round 2: %d candidates where C > B > A (consistent increase)" % len(candidates_abc))

    # Step 8: Stand still, take snapshot D
    print("[7/7] Standing still for 1s, taking snapshot D...")
    time.sleep(1.0)
    snap_d = {}
    for start, end in regions:
        snap_d.update(snapshot_region(sock, args.host, args.port, start, end, args.chunk))

    # Filter: D == C (value stable when not moving)
    final_candidates = []
    for addr, val_a, val_b, val_c, delta_ab, delta_bc in candidates_abc:
        if (addr + 1) not in snap_d:
            continue
        val_d = snap_d[addr] | (snap_d[addr + 1] << 8)
        delta_cd = val_d - val_c
        # Allow small drift (±2) for stability
        if abs(delta_cd) <= 2:
            final_candidates.append((addr, val_a, val_b, val_c, val_d, delta_ab, delta_bc, delta_cd))

    print("\n" + "=" * 80)
    print("RESULTS: %d candidate addresses for player X" % len(final_candidates))
    print("=" * 80)

    if not final_candidates:
        print("No stable candidates found. Showing Round 2 results instead:")
        for addr, val_a, val_b, val_c, dab, dbc in candidates_abc[:30]:
            print("  %06X: A=%5d B=%5d C=%5d  delta_AB=%+d delta_BC=%+d" %
                  (addr, val_a, val_b, val_c, dab, dbc))
    else:
        # Sort by consistency of deltas (prefer similar AB and BC deltas)
        final_candidates.sort(key=lambda x: abs(x[5] - x[6]))
        for addr, va, vb, vc, vd, dab, dbc, dcd in final_candidates:
            print("  %06X: A=%5d B=%5d C=%5d D=%5d  delta_AB=%+d delta_BC=%+d delta_CD=%+d" %
                  (addr, va, vb, vc, vd, dab, dbc, dcd))

        best = final_candidates[0]
        print("\nBest candidate: %06X" % best[0])
        print("  Use:  -e PLAYER_X_ADDR=%06X" % best[0])
        print("  Values: %d → %d → %d → %d (deltas: %+d, %+d, %+d)" %
              (best[1], best[2], best[3], best[4], best[5], best[6], best[7]))

    # Restore save state
    print("\nRestoring save state...")
    load_state(sock, args.host, args.port)
    sock.close()


if __name__ == "__main__":
    main()
