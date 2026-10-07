"""Make sure no benchmark sentence is also a training sentence.

Compares benchmark/benchmark.tsv against recording/prompts.tsv.
  FAIL  exact matches and near-duplicates (difflib ratio >= 0.8)
  WARN  names and places that appear in both files, so I can decide which
        benchmark names count as "seen" vs "unseen" in training.
        Roman script: capitalised words, ignoring "I". Devanagari has no capitals,
        so Devanagari words are transliterated and fuzzy-matched against the
        names found in the prompts (catches जयपुर ~ Jaipur).

Run:  python scripts/check_overlap.py
Exits 0 when there are no failures.
"""
import csv
import re
import string
import sys
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmark" / "benchmark.tsv"
PROMPTS = ROOT / "recording" / "prompts.tsv"
NEAR_DUP = 0.8
NAME_MATCH = 0.75  # transliterated Devanagari word vs a Roman name from the prompts

# string.punctuation is ASCII only, so add the Devanagari danda and curly quotes.
PUNCT = string.punctuation + "।॥“”‘’"


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"missing file: {path}")
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def normalise(text: str) -> str:
    text = text.lower().translate(str.maketrans("", "", PUNCT))
    return re.sub(r"\s+", " ", text).strip()


def capitalised_words(text: str, skip_initial: bool = True) -> set[str]:
    """Capitalised words, likely names and places. "I" is never a name.

    For prompts we skip sentence-initial words, so ordinary sentence starters
    ("Please", "Your") never enter the list of training names.
    """
    words = set()
    tokens = text.split()
    for i, tok in enumerate(tokens):
        word = tok.strip(PUNCT)
        if len(word) < 2 or not word[0].isupper() or word.isupper():  # skips "I" and acronyms/letters
            continue
        if skip_initial and (i == 0 or tokens[i - 1][-1] in ".?!"):
            continue
        words.add(word)
    return words


# Rough Devanagari -> Latin, only good enough to match names like जयपुर ~ jaipur.
CONSONANTS = dict(zip("कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह",
                      ["k", "kh", "g", "gh", "n", "ch", "chh", "j", "jh", "n", "t", "th", "d", "dh", "n",
                       "t", "th", "d", "dh", "n", "p", "ph", "b", "bh", "m", "y", "r", "l", "v", "sh", "sh", "s", "h"]))
VOWEL_SIGNS = dict(zip("ािीुूृेैोौ", ["a", "i", "i", "u", "u", "ri", "e", "ai", "o", "au"]))
VOWELS = dict(zip("अआइईउऊऋएऐओऔ", ["a", "a", "i", "i", "u", "u", "ri", "e", "ai", "o", "au"]))


def transliterate(word: str) -> str:
    out = []
    for i, ch in enumerate(word):
        nxt = word[i + 1] if i + 1 < len(word) else ""
        if ch in CONSONANTS:
            out.append(CONSONANTS[ch])
            if nxt not in VOWEL_SIGNS and nxt != "्" and nxt:  # inherent 'a', dropped at word end
                out.append("a")
        elif ch in VOWEL_SIGNS:
            out.append(VOWEL_SIGNS[ch])
        elif ch in VOWELS:
            out.append(VOWELS[ch])
        elif ch in "ंँ":
            out.append("n")
    return "".join(out)


def devanagari_name_matches(text: str, names: set[str]) -> set[str]:
    hits = set()
    for tok in text.split():
        word = tok.strip(PUNCT)
        if not word or not ("\u0900" <= word[0] <= "\u097f"):
            continue
        roman = transliterate(word)
        if len(roman) < 5:  # short common words (agar, vahi, shaam) match names by accident
            continue
        for name in names:
            if SequenceMatcher(None, roman, name.lower()).ratio() >= NAME_MATCH:
                hits.add(f"{word}~{name}")
    return hits


def main() -> None:
    import argparse

    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    bench = read_rows(BENCHMARK)
    prompts = read_rows(PROMPTS)
    prompt_norm = [(p["id"], p["text"], normalise(p["text"])) for p in prompts]

    failures = []
    for b in bench:
        b_norm = normalise(b["text"])
        for pid, ptext, p_norm in prompt_norm:
            matcher = SequenceMatcher(None, b_norm, p_norm)
            if matcher.real_quick_ratio() < NEAR_DUP or matcher.quick_ratio() < NEAR_DUP:
                continue
            ratio = matcher.ratio()
            if ratio >= NEAR_DUP:
                kind = "EXACT" if b_norm == p_norm else f"NEAR {ratio:.2f}"
                failures.append((kind, b["id"], b["text"], pid, ptext))

    prompt_caps: dict[str, list[str]] = {}
    for p in prompts:
        for w in capitalised_words(p["text"]):
            prompt_caps.setdefault(w, []).append(p["id"])
    names = set(prompt_caps)
    shared = sorted(
        {(w, b["id"]) for b in bench for w in capitalised_words(b["text"], skip_initial=False) if w in names}
        | {(w, b["id"]) for b in bench for w in devanagari_name_matches(b["text"], names)}
    )

    for kind, bid, btext, pid, ptext in failures:
        print(f"FAIL {kind}\n  {bid}: {btext}\n  {pid}: {ptext}")
    for w, bid in shared:
        ids = prompt_caps[w.split("~")[-1]]
        print(f"WARN '{w}' in {bid} also appears in training ({len(ids)}x, e.g. {ids[0]})")

    print(f"\n{len(bench)} benchmark rows vs {len(prompts)} prompts: "
          f"{len(failures)} failures, {len(shared)} warnings")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
