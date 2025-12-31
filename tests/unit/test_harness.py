"""Basic harness test to verify unit test discovery works."""

import unittest


class TestHarness(unittest.TestCase):
    """Sanity check that unittest discovery is working."""

    def test_harness_works(self):
        """Verify the test harness is functional."""
        self.assertTrue(True)

    def test_imports_work(self):
        """Verify we can import without loading heavy dependencies."""
        # These should always be safe to import
        import sys
        import os
        self.assertIn("sys", dir())


if __name__ == "__main__":
    unittest.main()

