#!/usr/bin/env python3
"""Probe candidate player X RAM addresses to find the right one for PLAYER_X_ADDR.

Usage:
    # With game running in Docker (default RetroArch UDP port 55355):
    python scripts/probe_player_x.py

    # Custom host/port:
    python scripts/probe_player_x.py --host 127.0.0.1 --port 55355

Reads candidate addresses every 0.5s and prints their values.
Move the character right/left and observe which address tracks movement.
Press Ctrl+C to stop.
"""
import argparse
import socket
import time


CANDIDATES = [
    ("02242630", 2, "screenX"),
    ("02242646", 2, "screenY"),
    ("022410CC", 2, "pixelX"),
]


def read_ram(sock, host, port, address, num_bytes, retries=3):
    for _ in range(retries):
        try:
            cmd = "READ_CORE_RAM %s %d\n" % (address, num_bytes)
            sock.sendto(cmd.encode(), (host, port))
            data, _ = sock.recvfrom(1024)
            parts = data.decode().strip().split()
            values = [int(b, 16) for b in parts[2:]]
            if len(values) == num_bytes:
                # 16-bit little-endian
                if num_bytes == 2:
                    return values[0] | (values[1] << 8)
                return values[0]
        except socket.timeout:
            continue
    return None


def main():
    parser = argparse.ArgumentParser(description="Probe player X RAM addresses")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=55355)
    parser.add_argument("--interval", type=float, default=0.5,
                        help="Seconds between reads (default: 0.5)")
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(0.1)

    prev = {}
    print("Probing RAM addresses every %.1fs. Move the character and watch for changes." % args.interval)
    print("Press Ctrl+C to stop.\n")
    header = "%-12s" % "addr"
    for addr, _, label in CANDIDATES:
        header += "  %-18s" % ("%s (%s)" % (label, addr))
    print(header)
    print("-" * len(header))

    try:
        while True:
            line = "%-12s" % time.strftime("%H:%M:%S")
            for addr, nbytes, label in CANDIDATES:
                val = read_ram(sock, args.host, args.port, addr, nbytes)
                prev_val = prev.get(addr)
                if val is not None:
                    delta = ""
                    if prev_val is not None:
                        d = val - prev_val
                        if d != 0:
                            delta = " (%+d)" % d
                    line += "  %-18s" % ("%d%s" % (val, delta))
                    prev[addr] = val
                else:
                    line += "  %-18s" % "TIMEOUT"
            print(line)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nDone.")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
