"""How often Whisper misses Indian names on my REAL recordings (no TTS involved).

For the English and Hinglish takes used in training, every capitalised word in
the prompt that isn't sentence-initial, an acronym, a title, a weekday or a
month counts as a name word (scripts/check_overlap.py's rule). A name word is
missed if it isn't in Whisper's transcript of that take (data/processed/asr.csv,
the data-prep Whisper run). Prints counts only, because the prompts are private.

Usage:  python eval/names_real.py [--prompts recording/prompts.tsv] [--processed data/processed]
"""
import argparse
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from check_overlap import capitalised_words  # noqa: E402

NOT_NAMES = {"Mr", "Mrs", "Ms", "Dr", "Sir", "Madam", "Ji", "Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday", "January", "February", "March", "April", "May", "June", "July",
             "August", "September", "October", "November", "December", "Namaste", "Diwali", "Holi", "Saturdays", "Seventeen", "Road", "Ring",
             "Outer", "Lake", "Sector", "Salt", "Fancy", "Bazaar", "Pin", "WhatsApp", "Aadhaar", "Puja"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", type=Path, default=ROOT / "recording" / "prompts.tsv")
    ap.add_argument("--processed", type=Path, default=ROOT / "data" / "processed")
    args = ap.parse_args()
    text = {r["id"]: r["text"] for r in csv.DictReader(open(args.prompts, encoding="utf-8"), delimiter="\t")}
    used = {r["id"]: r["lang"] for r in csv.DictReader(open(args.processed / "clips.csv", encoding="utf-8"))
            if r["status"] != "drop"}
    hyp = {r["id"]: r["hypothesis"] for r in csv.DictReader(open(args.processed / "asr.csv", encoding="utf-8"))}
    clips = words = missed = 0
    for i, lang in used.items():
        if i not in hyp or i not in text or not re.search(r"[A-Za-z]", text[i]):
            continue
        names = [w for w in capitalised_words(text[i]) if w not in NOT_NAMES and not w.isupper()]
        if not names:
            continue
        heard = set(re.findall(r"\w+", hyp[i].lower()))
        clips += 1
        for name in names:
            for w in re.findall(r"\w+", name.lower()):
                words += 1
                missed += w not in heard
    print(f"real takes with names: {clips} clips, name words missed by Whisper: {missed}/{words} = {missed / words:.2f}")


if __name__ == "__main__":
    main()
