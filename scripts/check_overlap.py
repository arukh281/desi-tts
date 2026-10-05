"""Make sure no benchmark sentence is also a training sentence.

Compares benchmark/benchmark.tsv against recording/prompts.tsv.
  FAIL  exact matches and near-duplicates (difflib ratio >= 0.8)
  WARN  capitalised words (names, places) that appear in both files, so I can
        decide which benchmark names count as "seen" vs "unseen" in training

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


def capitalised_words(text: str) -> set[str]:
    """Capitalised words that don't start a sentence: likely names and places."""
    words = set()
    tokens = text.split()
    for i, tok in enumerate(tokens):
        word = tok.strip(PUNCT)
        if not word or not word[0].isupper():
            continue
        if i == 0 or tokens[i - 1][-1] in ".?!":
            continue
        words.add(word)
    return words


def main() -> None:
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
    shared = sorted(
        {(w, b["id"]) for b in bench for w in capitalised_words(b["text"]) if w in prompt_caps}
    )

    for kind, bid, btext, pid, ptext in failures:
        print(f"FAIL {kind}\n  {bid}: {btext}\n  {pid}: {ptext}")
    for w, bid in shared:
        ids = prompt_caps[w]
        print(f"WARN '{w}' in {bid} also appears in training ({len(ids)}x, e.g. {ids[0]})")

    print(f"\n{len(bench)} benchmark rows vs {len(prompts)} prompts: "
          f"{len(failures)} failures, {len(shared)} warnings")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
