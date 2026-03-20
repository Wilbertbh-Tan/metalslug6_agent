import unittest

from src.env.ram_decode import decode_bcd_score


class DecodeBcdScoreTests(unittest.TestCase):
    def test_returns_none_for_missing_or_short_data(self):
        self.assertIsNone(decode_bcd_score(None))
        self.assertIsNone(decode_bcd_score([]))
        self.assertIsNone(decode_bcd_score([0x12, 0x34, 0x56]))

    def test_decodes_valid_bcd_bytes(self):
        # 5 bytes: bytes 0-3 BCD, byte 4 = 10,000s. Score = byte4*10000 + bcd_part.
        # 0x00 0x00 0x12 0x34 0x00 -> bcd_part=1234, upper=0 -> 1234
        self.assertEqual(decode_bcd_score([0x00, 0x00, 0x12, 0x34, 0x00]), 1234)
        # 0x00 0x00 0x00 0x01 0x00 -> bcd_part=1, upper=0 -> 1
        self.assertEqual(decode_bcd_score([0x00, 0x00, 0x00, 0x01, 0x00]), 1)
        # 0x00 0x00 0x11 0x60 0x03 -> bcd_part=1160, upper=3 -> 31160 (on-screen score)
        self.assertEqual(decode_bcd_score([0x00, 0x00, 0x11, 0x60, 0x03]), 31160)

    def test_rejects_invalid_bcd_nibbles(self):
        self.assertIsNone(decode_bcd_score([0x00, 0x0A, 0x12, 0x34, 0x00]))
        self.assertIsNone(decode_bcd_score([0x00, 0x00, 0x1F, 0x34, 0x00]))


if __name__ == "__main__":
    unittest.main()
