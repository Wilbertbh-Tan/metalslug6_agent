import socket
import time

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.settimeout(1.0)

SCORE_ADDR = "003869BC"


def read_raw(address, num_bytes):
    try:
        cmd = f"READ_CORE_RAM {address} {num_bytes}\n"
        sock.sendto(cmd.encode(), ("127.0.0.1", 55355))
        data, _ = sock.recvfrom(1024)
        response = data.decode().strip()
        parts = response.split()
        return [int(b, 16) for b in parts[2:]]
    except socket.timeout:
        print("Timeout!")
        return []


def decode_le16(b):
    """Little-endian 16-bit"""
    return b[0] | (b[1] << 8) if len(b) >= 2 else 0


def decode_le32(b):
    """Little-endian 32-bit"""
    return b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24) if len(b) >= 4 else 0


def decode_be16(b):
    """Big-endian 16-bit"""
    return (b[0] << 8) | b[1] if len(b) >= 2 else 0


def decode_be32(b):
    """Big-endian 32-bit"""
    return (b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3] if len(b) >= 4 else 0


def decode_bcd(b):
    """BCD decode (each nibble is a digit)"""
    result = 0
    for byte in reversed(b):
        hi = (byte >> 4) & 0xF
        lo = byte & 0xF
        result = result * 100 + hi * 10 + lo
    return result


print("Reading score in multiple formats. Kill enemies and watch which one tracks.")
print("Press Ctrl+C to stop.\n")
print(
    f"{'Raw bytes':<20} {'LE16':<10} {'LE32':<10} {'BE16':<10} {'BE32':<10} {'BCD':<10}"
)
print("-" * 70)

for i in range(30):
    b = read_raw(SCORE_ADDR, 4)
    if b:
        raw = " ".join(f"{x:02X}" for x in b)
        print(
            f"{raw:<20} {decode_le16(b):<10} {decode_le32(b):<10} {decode_be16(b):<10} {decode_be32(b):<10} {decode_bcd(b):<10}"
        )
    time.sleep(1)

sock.close()
