"""End-to-end test of data prep on a synthetic session where every clip has one known problem.

Run:  python -m unittest discover -s tests -v
"""
import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

EXPECTED = {
    "syn_clean": ("keep", None),
    "syn_click": ("keep", None),  # clicks are removed (see edge_clicks_removed), not flagged
    "syn_clip": ("drop", "clipping"),
    "syn_long": ("drop", "too_long"),
    "syn_noisy": ("drop", "low_snr"),
    "syn_pause": ("drop", "long_pause"),
    "syn_quiet": ("drop", "too_quiet"),
    "syn_wrong": ("drop", "transcript_mismatch"),
}


class SyntheticSession(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        t = Path(cls.tmp.name)
        run = lambda *a: subprocess.run([PY, *a], check=True, capture_output=True, text=True)
        run(str(ROOT / "data_prep" / "make_synthetic.py"), "--out", str(t))
        run(str(ROOT / "data_prep" / "prepare.py"), "--raw", str(t / "raw"),
            "--prompts", str(t / "prompts.tsv"), "--out", str(t / "processed"))
        # Stand-in for the Whisper check: every clip is "heard" correctly except
        # syn_wrong. finalize.py re-scores CER from these hypotheses itself.
        with open(t / "prompts.tsv", encoding="utf-8") as f:
            texts = {r["id"]: r["text"] for r in csv.DictReader(f, delimiter="\t")}
        with open(t / "processed" / "asr.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "hypothesis", "cer"])
            for clip in EXPECTED:
                hyp = "the weather is nice today" if clip == "syn_wrong" else texts[clip]
                w.writerow([clip, hyp, ""])
        run(str(ROOT / "data_prep" / "finalize.py"), "--processed", str(t / "processed"))
        with open(t / "processed" / "keep_drop.csv", encoding="utf-8") as f:
            cls.result = {r["id"]: r for r in csv.DictReader(f)}
        cls.out = t / "processed"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_every_check_fires_with_the_right_reason(self):
        for clip, (status, reason) in EXPECTED.items():
            with self.subTest(clip=clip):
                self.assertEqual(self.result[clip]["status"], status, self.result[clip]["reasons"])
                if reason:
                    self.assertIn(reason, self.result[clip]["reasons"])

    def test_edge_clicks_are_removed_and_counted(self):
        with open(self.out / "clips.csv", encoding="utf-8") as f:
            clips = {r["id"]: r for r in csv.DictReader(f)}
        self.assertEqual(clips["syn_click"]["edge_clicks_removed"], "2")

    def test_every_drop_and_flag_has_a_reason(self):
        for row in self.result.values():
            if row["status"] != "keep":
                self.assertTrue(row["reasons"], row["id"])

    def test_training_files_are_22khz_and_metadata_is_coqui_format(self):
        import soundfile as sf

        self.assertEqual(sf.info(self.out / "wavs" / "syn_clean.wav").samplerate, 22050)
        header = (self.out / "metadata_en_train.csv").read_text().splitlines()[0]
        self.assertEqual(header, "audio_file|text|speaker_name")

    def test_dropped_clips_are_not_in_metadata(self):
        text = "".join((self.out / f"metadata_{l}_{p}.csv").read_text()
                       for l in ("en", "hi") for p in ("train", "val"))
        self.assertNotIn("syn_clip", text)
        self.assertNotIn("syn_wrong", text)


if __name__ == "__main__":
    unittest.main()
