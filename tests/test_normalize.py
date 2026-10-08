"""Tests for text/normalize.py: one per rule, every benchmark expected form, and prompts unchanged.

Run:  python -m pytest tests/test_normalize.py   (or python -m unittest discover -s tests)
"""
import csv
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from text.normalize import normalize  # noqa: E402


def say(text, lang="en"):
    return normalize(text, lang)[0]


class English(unittest.TestCase):
    def test_currency_indian_grouping(self):
        self.assertEqual(say("₹1,25,000"), "one lakh twenty five thousand rupees")
        self.assertEqual(say("₹12,750"), "twelve thousand seven hundred and fifty rupees")
        self.assertEqual(say("₹1.2 crore"), "one point two crore rupees")

    def test_currency_paise_and_rs(self):
        self.assertEqual(say("Rs. 4,999.50"), "four thousand nine hundred and ninety nine rupees and fifty paise")

    def test_numbers_and_ordinals(self):
        self.assertEqual(say("3 missed payments"), "three missed payments")
        self.assertEqual(say("the 3rd attempt by the 20th"), "the third attempt by the twentieth")

    def test_times(self):
        self.assertEqual(say("4:30 PM"), "four thirty in the afternoon")
        self.assertEqual(say("9:30 AM to 4 PM"), "nine thirty in the morning to four in the afternoon")
        self.assertEqual(say("7:40 PM"), "seven forty in the evening")
        self.assertEqual(say("by 11:45 tonight"), "by eleven forty five tonight")

    def test_dates_dd_mm(self):
        self.assertEqual(say("03/10/2026"), "the third of October twenty twenty six")
        self.assertEqual(say("15th August"), "the fifteenth of August")
        self.assertEqual(say("31 Dec 2026"), "the thirty first of December twenty twenty six")

    def test_digit_strings(self):
        self.assertEqual(say("OTP 482913"), "O T P four eight two nine one three")
        self.assertEqual(say("9876543210"), "nine eight seven six five four three two one zero")
        self.assertEqual(say("ending in 4421"), "ending in four four two one")

    def test_decimals_and_percent(self):
        self.assertEqual(say("1.5 GB"), "one point five G B")
        self.assertEqual(say("10.5%"), "ten point five per cent")

    def test_acronyms_spelled_like_the_prompts(self):
        self.assertEqual(say("PAN and PIN"), "P A N and P I N")  # prompts.tsv spells both out
        self.assertEqual(say("I am OK"), "I am O K")  # single "I" is left alone

    def test_asr_style_numbers(self):
        # how Whisper writes things back, so CER compares like with like
        self.assertEqual(say("5000 rupees"), "five thousand rupees")
        self.assertEqual(say("Rs 2,000"), "two thousand rupees")
        self.assertEqual(say("1,82,400 rupees"), "one lakh eighty two thousand four hundred rupees")
        self.assertEqual(say("at 10.30 in the morning"), "at ten thirty in the morning")
        self.assertEqual(say("at 7.40 pm"), "at seven forty in the evening")
        self.assertEqual(say("10,000 रुपे", "hi"), "दस हज़ार रुपये")

    def test_rewrites_are_listed(self):
        text, rewrites = normalize("Your EMI of ₹50 is due.", "en")
        self.assertEqual(rewrites, [("₹50", "fifty rupees"), ("EMI", "E M I")])
        self.assertEqual(text, "Your E M I of fifty rupees is due.")

    def test_deterministic(self):
        self.assertEqual(normalize("₹2,340 at 4:30 PM", "en"), normalize("₹2,340 at 4:30 PM", "en"))


class Hinglish(unittest.TestCase):
    def test_hindi_number_words_in_roman(self):
        self.assertEqual(say("₹8,500", "hinglish"), "aath hazaar paanch sau rupaye")
        self.assertEqual(say("28 din", "hinglish"), "atthaais din")

    def test_baje_times(self):
        self.assertEqual(say("4 baje", "hinglish"), "chaar baje")
        self.assertEqual(say("6:30 baje", "hinglish"), "saadhe chhe baje")
        self.assertEqual(say("1:30 baje", "hinglish"), "dedh baje")
        self.assertEqual(say("2:45 baje", "hinglish"), "paune teen baje")

    def test_digits_and_call_numbers(self):
        self.assertEqual(say("number 77812", "hinglish"), "number saat saat aath ek do")
        self.assertEqual(say("aap 198 pe call kariye", "hinglish"), "aap ek nau aath pe call kariye")

    def test_date(self):
        self.assertEqual(say("15/11/2026", "hinglish"), "pandrah November do hazaar chhabbees")


class Hindi(unittest.TestCase):
    def test_currency(self):
        self.assertEqual(say("₹1,50,000", "hi"), "एक लाख पचास हज़ार रुपये")

    def test_times(self):
        self.assertEqual(say("10:30 बजे", "hi"), "साढ़े दस बजे")
        self.assertEqual(say("9:15 बजे", "hi"), "सवा नौ बजे")
        self.assertEqual(say("7:45 बजे", "hi"), "पौने आठ बजे")
        self.assertEqual(say("2:30 बजे", "hi"), "ढाई बजे")

    def test_latin_acronym_as_devanagari_letters(self):
        self.assertEqual(say("OTP", "hi"), "ओ टी पी")

    def test_date_and_digits(self):
        self.assertEqual(say("31/03/2027", "hi"), "इकतीस मार्च दो हज़ार सत्ताईस")
        self.assertEqual(say("40917", "hi"), "चार ज़ीरो नौ एक सात")


class Data(unittest.TestCase):
    def test_every_benchmark_expected_form(self):
        with open(ROOT / "benchmark" / "benchmark.tsv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        for row in rows:
            out = say(row["text"], row["lang"])
            for part in filter(None, row["check"].split("; ")):
                if " = " in part:
                    expected = re.sub(r"\s*\(DD/MM\)", "", part.split(" = ", 1)[1])
                    with self.subTest(id=row["id"], expected=expected):
                        self.assertIn(expected, out)

    def test_prompts_are_unchanged(self):
        path = ROOT / "recording" / "prompts.tsv"
        if not path.exists():
            self.skipTest("prompts.tsv is private and not present here")
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                lang = "hi" if row["lang"] == "hi" else "en"
                with self.subTest(id=row["id"]):
                    self.assertEqual(say(row["text"], lang), row["text"])


if __name__ == "__main__":
    unittest.main()
