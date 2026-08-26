import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from pocket_tts_mlx.data.audio import audio_read, stream_audio_chunks


class AudioEdgeCaseTests(unittest.TestCase):
    def test_audio_read_uses_soundfile_for_8_bit_wav(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "silence_u8.wav"
            with wave.open(str(output_file), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(1)
                wav_file.setframerate(8000)
                wav_file.writeframes(bytes([128] * 16))

            audio, sample_rate = audio_read(output_file)

        self.assertEqual(sample_rate, 8000)
        self.assertEqual(audio.shape, (1, 16))
        np.testing.assert_allclose(audio, np.zeros_like(audio))

    def test_stream_audio_chunks_patches_seekable_wav_header(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "output.wav"
            samples = np.zeros(24000, dtype=np.float32)

            stream_audio_chunks(output_file, iter([samples]), sample_rate=24000)

            with wave.open(str(output_file), "rb") as wav_file:
                self.assertEqual(wav_file.getframerate(), 24000)
                # The writer intentionally appends 200 ms of silence.
                self.assertEqual(wav_file.getnframes(), 28800)


if __name__ == "__main__":
    unittest.main()
