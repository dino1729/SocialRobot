"""Integration tests for SocialRobot audio pipeline.

These tests require real TTS/STT models and take longer to run.
They are skipped by default unless explicitly enabled.

Run integration tests with:
    INTEGRATION_TESTS=1 python -m unittest discover -s tests/integration -p 'test_*.py' -v

Or run specific engine tests:
    INTEGRATION_TESTS=1 python -m unittest tests.integration.test_kokoro_stt -v
"""

import os

# Check if integration tests should run
INTEGRATION_TESTS_ENABLED = os.environ.get('INTEGRATION_TESTS', '').lower() in ('1', 'true', 'yes')
SKIP_REASON = "Integration tests disabled. Set INTEGRATION_TESTS=1 to run."

