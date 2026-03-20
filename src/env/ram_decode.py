"""Utilities for decoding RAM values from RetroArch UDP reads."""

from __future__ import annotations


def decode_bcd_score(raw_bytes: list[int] | bytes | bytearray | None) -> int | None:
    """Decode 5-byte little-endian BCD score used by Metal Slug 6.

    Score is at 003869BC (5 bytes). Bytes are in little-endian BCD order:
    the most significant digits are in the higher-addressed bytes.

    Reading byte4, byte3, byte2 and concatenating their BCD digit pairs
    gives the on-screen score. Bytes 0-1 are always 00 in normal gameplay.

    Example: raw = [00, 00, 80, 13, 04]
             byte4=04 byte3=13 byte2=80 → "04" || "13" || "80" → 41380
    """
    if not raw_bytes or len(raw_bytes) < 5:
        return None

    # Decode bytes 2, 3, 4 in reverse order (little-endian BCD)
    result = 0
    for byte in (raw_bytes[4], raw_bytes[3], raw_bytes[2]):
        hi = (byte >> 4) & 0xF
        lo = byte & 0xF
        if hi > 9 or lo > 9:
            return None
        result = result * 100 + hi * 10 + lo

    return result
