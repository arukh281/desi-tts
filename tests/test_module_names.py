"""Scripts run from their own folder, so a file named like a stdlib module shadows it.

infer/profile.py once broke every synthesis run on Kaggle: transformers imports
the stdlib `profile` module and got ours instead.

Run:  python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class NoStdlibShadowing(unittest.TestCase):
    def test_no_script_named_like_a_stdlib_module(self):
        clashes = [str(p.relative_to(ROOT)) for folder in ("infer", "eval", "train", "data_prep", "recording",
                                                            "scripts", "text", "kaggle")
                   for p in (ROOT / folder).glob("*.py") if p.stem in sys.stdlib_module_names]
        self.assertEqual(clashes, [])


if __name__ == "__main__":
    unittest.main()
