"""Tests for scripts/check_overlap.py name detection.

Run:  python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import check_overlap as co  # noqa: E402


class Names(unittest.TestCase):
    def test_pronoun_i_and_acronyms_are_not_names(self):
        words = co.capitalised_words("Give me a moment while I pull up your KYC, Anjali.", skip_initial=False)
        self.assertEqual(words, {"Give", "Anjali"})

    def test_prompts_skip_sentence_starters(self):
        self.assertEqual(co.capitalised_words("Please call Pune. Your order is late."), {"Pune"})

    def test_devanagari_name_matches_roman_name(self):
        self.assertTrue(co.devanagari_name_matches("आपका पार्सल जयपुर से निकल चुका है।", {"Jaipur"}))
        self.assertTrue(co.devanagari_name_matches("वेंकटेश जी, नमस्ते।", {"Venkatesh"}))

    def test_short_common_hindi_words_do_not_match(self):
        self.assertFalse(co.devanagari_name_matches("अगर वही शाम को", {"Nagar", "Vashi", "Sharma"}))


if __name__ == "__main__":
    unittest.main()
