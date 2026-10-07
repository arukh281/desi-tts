"""Tests for the non-audio logic in recording/record.py and recording/order.py.

Run:  python -m unittest discover -s tests -v
"""
import csv
import sys
import tempfile
import unittest
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "recording"))
sys.path.insert(0, str(ROOT / "scripts"))

import order  # noqa: E402
import record  # noqa: E402
from check_overlap import normalise  # noqa: E402

SR = record.SR


def tone(amplitude: float, seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype("float32")


class CheckTake(unittest.TestCase):
    def test_normal_take_has_no_warnings(self):
        self.assertEqual(record.check_take(tone(0.3, 3)), [])

    def test_clipping(self):
        self.assertIn("clipped", record.check_take(np.clip(tone(1.5, 3), -1, 1))[0])

    def test_quiet(self):
        self.assertIn("low level", record.check_take(tone(0.01, 3))[0])

    def test_pure_silence_points_to_mic_permission(self):
        self.assertIn("Privacy", record.check_take(np.zeros(SR * 2, dtype="float32"))[0])

    def test_too_long(self):
        self.assertTrue(any("over" in w for w in record.check_take(tone(0.3, 12))))


class Meter(unittest.TestCase):
    def test_meter_ends(self):
        self.assertTrue(record.meter_bar(-90).startswith("[" + "." * 30))
        self.assertIn("CLIP", record.meter_bar(0.0))

    def test_device_check(self):
        self.assertTrue(record.is_built_in("MacBook Air Microphone"))
        self.assertFalse(record.is_built_in("Master Buds Max"))


class Resume(unittest.TestCase):
    def test_skips_prompts_recorded_in_any_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)
            record.save_take(raw / "session_1" / "p0001.wav", tone(0.3, 1))
            record.save_take(raw / "session_2" / "p0003.wav", tone(0.3, 1))
            prompts = [{"id": f"p000{i}"} for i in range(1, 5)]
            left = record.todo(prompts, record.recorded_ids(raw))
            self.assertEqual([p["id"] for p in left], ["p0002", "p0004"])


class Tags(unittest.TestCase):
    def tags(self, text, lang="en"):
        return order.tags_for({"id": "x", "lang": lang, "text": text})

    def test_currency_numbers_acronyms(self):
        t = self.tags("Your E M I of twelve thousand rupees is due tomorrow.")
        for tag in ("numbers", "currency", "acronyms", "dates_times"):
            self.assertIn(tag, t)

    def test_names_and_places(self):
        t = self.tags("Please meet Priya at the Indiranagar branch.")
        self.assertIn("names", t)
        self.assertIn("places", t)

    def test_sentence_start_is_not_a_name(self):
        self.assertNotIn("names", self.tags("Thanks for calling today."))

    def test_language_groups(self):
        self.assertEqual(order.language_group({"lang": "en", "text": "Aapka payment ho gaya hai."}), "hinglish")
        self.assertEqual(order.language_group({"lang": "en", "text": "Your payment is done."}), "en")
        self.assertEqual(order.language_group({"lang": "hi", "text": "भुगतान हो गया है।"}), "hi")


class OrderLogic(unittest.TestCase):
    def test_hard_first_within_group(self):
        rows = [
            {"id": "easy", "lang": "en", "text": "Have a nice day."},
            {"id": "hard", "lang": "en", "text": "Priya, your E M I of five thousand rupees is due tomorrow."},
        ]
        self.assertEqual([r["id"] for r in order.build_order(rows)], ["hard", "easy"])

    def test_interleave_spreads_groups_evenly(self):
        out = order.interleave({"a": list("aaaaaa"), "b": list("bb")})
        self.assertEqual(len(out), 8)
        self.assertIn("b", out[:4])  # the small group isn't pushed to the end

    def test_too_long(self):
        self.assertEqual(order.too_long("short sentence"), [])
        self.assertTrue(order.too_long("word " * 50))


class ReferenceSentences(unittest.TestCase):
    """The reference clips must not reuse training or benchmark text."""

    def other_texts(self):
        texts = []
        for path in (ROOT / "recording" / "prompts.tsv", ROOT / "benchmark" / "benchmark.tsv"):
            if path.exists():
                with open(path, encoding="utf-8", newline="") as f:
                    texts += [r["text"] for r in csv.DictReader(f, delimiter="\t")]
        return texts

    def test_no_overlap(self):
        with open(ROOT / "recording" / "reference.txt", encoding="utf-8", newline="") as f:
            refs = [r["text"] for r in csv.DictReader(f, delimiter="\t")]
        self.assertEqual(len(refs), 3)
        others = self.other_texts()
        if not others:
            self.skipTest("prompts.tsv is private and not present here")
        for ref in refs:
            for other in others:
                ratio = SequenceMatcher(None, normalise(ref), normalise(other)).ratio()
                self.assertLess(ratio, 0.8, f"{ref!r} too close to {other!r}")


if __name__ == "__main__":
    unittest.main()
