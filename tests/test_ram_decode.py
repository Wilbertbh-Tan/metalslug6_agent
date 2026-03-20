import unittest

from src.env.ram_decode import decode_bcd_score


class DecodeBcdScoreTests(unittest.TestCase):
    def test_returns_none_for_missing_or_short_data(self):
        self.assertIsNone(decode_bcd_score(None))
        self.assertIsNone(decode_bcd_score([]))
        self.assertIsNone(decode_bcd_score([0x12, 0x34, 0x56]))

    def test_decodes_valid_bcd_bytes(self):
        # 0x00 0x00 0x12 0x34 -> 1234 * 100
        self.assertEqual(decode_bcd_score([0x00, 0x00, 0x12, 0x34]), 123400)
        self.assertEqual(decode_bcd_score([0x00, 0x00, 0x00, 0x01]), 100)

    def test_rejects_invalid_bcd_nibbles(self):
        self.assertIsNone(decode_bcd_score([0x00, 0x0A, 0x12, 0x34]))
        self.assertIsNone(decode_bcd_score([0x00, 0x00, 0x1F, 0x34]))


if __name__ == "__main__":
    unittest.main()
