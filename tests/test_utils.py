import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pocket_tts_mlx.utils import utils


class DownloadTests(unittest.TestCase):
    def test_http_cache_suffix_ignores_query_string(self):
        class Response:
            content = b"data"

            @staticmethod
            def raise_for_status():
                return None

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            with (
                patch.object(utils, "make_cache_directory", return_value=cache_dir),
                patch.object(utils.requests, "get", return_value=Response()),
            ):
                cached_file = utils.download_if_necessary(
                    "https://example.com/audio.wav?download=true"
                )

            self.assertEqual(cached_file.parent, cache_dir)
            self.assertEqual(cached_file.suffix, ".wav")
            self.assertNotIn("?", cached_file.name)
            self.assertEqual(cached_file.read_bytes(), b"data")


if __name__ == "__main__":
    unittest.main()
