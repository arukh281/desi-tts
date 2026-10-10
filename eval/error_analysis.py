"""Which kinds of words does a system get wrong? No GPU; runs on eval outputs.

Aligns each reference (spoken form) with Whisper's transcript word by word and
puts every missed or substituted reference word into one type:
  number   number words (incl. Hindi/Hinglish number words) from the normaliser
  acronym  spelled-out letters (K Y C, ओ टी पी)
  name     words of a name/place in the benchmark check column
  hindi    Devanagari words (Hindi rows)
  hinglish Roman Hindi words (Hinglish rows)
  english  everything else in English rows
With --listening, it also counts the pairs where listeners marked a wrong word,
by benchmark category.

Usage:
  python eval/error_analysis.py --synth-dir RUN/bench_norm_on --benchmark benchmark/benchmark.tsv \\
      [--listening KEY.json RESULTS.txt --system E3]
"""
import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
from asr_common import score_text  # noqa: E402
from text.normalize import EN_ONES, EN_TENS, HI_DEV, HI_ROMAN, LETTERS_HI  # noqa: E402

NUMBER_WORDS = set(EN_ONES) | set(EN_TENS) | set(HI_ROMAN) | set(HI_DEV) | {
    "hundred", "thousand", "lakh", "crore", "point", "first", "second", "third", "twentieth", "fifteenth",
    "thirty", "sau", "hazaar", "हज़ार", "हजार", "लाख", "सौ", "ज़ीरो", "zero", "saadhe", "साढ़े", "सवा", "पौने", "रुपये",
    "rupees", "rupaye", "paise"}
DEVANAGARI_LETTERS = {v.replace("़", "") for v in LETTERS_HI.values()} | set(LETTERS_HI.values())


def word_type(word: str, lang: str, names: set[str]) -> str:
    if word in names:
        return "name"
    if word in NUMBER_WORDS or re.fullmatch(r"\w*(ty|teen|th)", word) and word in NUMBER_WORDS:
        return "number"
    if (len(word) == 1 and word.isalpha() and word.isascii()) or word in DEVANAGARI_LETTERS:
        return "acronym"
    if lang == "hi":
        return "hindi"
    if lang == "hinglish":
        return "hinglish"
    return "english"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synth-dir", required=True)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--listening", nargs=2, metavar=("KEY", "RESULTS"))
    ap.add_argument("--system", default="E3")
    args = ap.parse_args()

    bench = {r["id"]: r for r in csv.DictReader(open(args.benchmark, encoding="utf-8"), delimiter="\t")}
    rows = list(csv.DictReader(open(Path(args.synth_dir) / "asr_eval.csv", encoding="utf-8")))
    by_type, examples, ref_counts = Counter(), defaultdict(list), Counter()
    for r in rows:
        lang = r["lang"]
        names = set()
        for part in bench[r["id"]]["check"].split("; "):
            if part and "=" not in part:
                names |= set(score_text(part, "hi" if lang == "hi" else "en").split())
        ref, hyp = r["reference"].split(), r["hypothesis"]
        hyp = score_text(hyp, "hi" if r.get("metric") == "hinglish_deva_rough" else lang).split()
        for w in ref:
            ref_counts[word_type(w, lang, names)] += 1
        for tag, i1, i2, j1, j2 in SequenceMatcher(None, ref, hyp, autojunk=False).get_opcodes():
            if tag in ("replace", "delete"):
                for w in ref[i1:i2]:
                    t = word_type(w, lang, names)
                    by_type[t] += 1
                    if len(examples[t]) < 6:
                        examples[t].append(f"{r['id']}: {w} -> {' '.join(hyp[j1:j2]) or '(missing)'}")

    print(f"### Whisper word errors on {args.synth_dir} ({len(rows)} rows)\n")
    print("| type | wrong words | of all words of that type | rate |\n|---|---|---|---|")
    for t, n in by_type.most_common():
        print(f"| {t} | {n} | {ref_counts[t]} | {n / max(1, ref_counts[t]):.2f} |")
    for t, _ in by_type.most_common(3):
        print(f"\n{t} examples:\n  " + "\n  ".join(examples[t]))

    if args.listening:
        key = json.load(open(args.listening[0]))["pairs"]
        marks = Counter()
        for line in open(args.listening[1], encoding="utf-8"):
            m = re.match(r"(P\d\d) nat=\S like=\S wa=(\S) wb=(\S)", line.strip())
            if not m or key[m[1]]["repeat_of"]:
                continue
            pair = key[m[1]]
            side = "A" if pair["A"].startswith(args.system) else "B" if pair["B"].startswith(args.system) else None
            if side and (m[2] if side == "A" else m[3]) == "Y":
                marks[bench[pair["row"]]["category"] + "/" + bench[pair["row"]]["lang"]] += 1
        print(f"\n### Listening: pairs where {args.system} was marked as having a wrong word, by category\n")
        for cat, n in marks.most_common():
            print(f"- {cat}: {n}")


if __name__ == "__main__":
    main()
