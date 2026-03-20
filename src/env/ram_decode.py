"""Utilities for decoding RAM values from RetroArch UDP reads."""

from __future__ import annotations


def decode_bcd_score(raw_bytes: list[int] | bytes | bytearray | None) -> int | None:
    """Decode 4-byte packed BCD score used by Metal Slug.

    The current project convention treats bytes in received order and scales by 100,
    matching the existing environment behavior.
    """
    if not raw_bytes or len(raw_bytes) < 4:
        return None

    result = 0
    for byte in raw_bytes:
        hi = (byte >> 4) & 0xF
        lo = byte & 0xF
        if hi > 9 or lo > 9:
            # Invalid BCD nibble indicates transient/garbage RAM read.
            return None
        result = result * 100 + hi * 10 + lo
    return result * 100
