import tempfile
import unittest
from pathlib import Path

from backend.services.dubbing import run_log


class RunLogTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_path = run_log.LOG_PATH
        run_log.LOG_PATH = Path(self.tmp.name) / "dubbing_runs.csv"

    def tearDown(self):
        run_log.LOG_PATH = self.old_path
        self.tmp.cleanup()

    def test_tts_batch_size_column_is_written(self):
        run_id = run_log.create_run(
            video_id="abc",
            duration_min=1.5,
            mode="omnivoice",
            asr_engine="faster:medium.en:cuda/float16",
        )
        run_log.update_run(run_id, tts_engine="omnivoice", tts_batch_size=4)

        row = run_log.get_run(run_id)
        self.assertIsNotNone(row)
        self.assertEqual(row["tts_batch_size"], "4")

        header = run_log.LOG_PATH.read_text(encoding="utf-8").splitlines()[0].split(",")
        self.assertIn("tts_batch_size", header)


if __name__ == "__main__":
    unittest.main()
