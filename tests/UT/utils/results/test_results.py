import unittest
import tempfile
import os
import json
import fcntl
from unittest.mock import patch, mock_open

from ais_bench.benchmark.utils.results.results import safe_write


class TestResults(unittest.TestCase):
    """Tests for results.py functions."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test_results.jsonl")

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)

    def test_safe_write(self):
        """Test safe_write function with file locking."""
        results_dict = {
            "task1": {"result": "success", "score": 0.95},
            "task2": {"result": "success", "score": 0.87}
        }

        safe_write(results_dict, self.test_file)

        # Verify file was created and contains expected content
        self.assertTrue(os.path.exists(self.test_file))
        with open(self.test_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 2)
            # Verify JSON content
            result1 = json.loads(lines[0].strip())
            result2 = json.loads(lines[1].strip())
            self.assertIn("result", result1)
            self.assertIn("result", result2)

    def test_safe_write_empty_dict(self):
        """Test safe_write with empty dictionary."""
        safe_write({}, self.test_file)
        self.assertTrue(os.path.exists(self.test_file))
        with open(self.test_file, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertEqual(content, "")


if __name__ == "__main__":
    unittest.main()

