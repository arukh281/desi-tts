"""Intelligibility: transcribe synthesised audio with Whisper and score CER/WER.

Reference = the expected spoken form of the input text (text/normalize.py,
which reproduces the benchmark's check column). Hypothesis = Whisper's
transcript, normalised the same way. Language forced per row (see asr_common.py).

Input is a folder from infer/synthesize.py (manifest.csv + <id>.wav).
Writes <synth-dir>/asr_eval.csv (per row) and prints mean CER/WER per category
and language. Category comes from --benchmark when the id is in it.

Hinglish: Whisper forced to English translates Hinglish instead of
transcribing it, so that score means nothing. With --deva FILE (id, text_deva),
Hinglish rows are decoded with language "hi" and compared in Devanagari
against the Devanagari version of the row. It's a rough signal (Whisper may
still write English loanwords in Latin script); listening is primary.

Usage (on a GPU):
  python eval/asr_eval.py --synth-dir outputs/synth
  python eval/asr_eval.py --synth-dir out/synth --benchmark benchmark/benchmark.tsv
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

from asr_common import cer, load_whisper, score_text, transcribe, wer

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synth-dir", required=True)
    ap.add_argument("--benchmark", default=str(ROOT / "benchmark" / "benchmark.tsv"))
    ap.add_argument("--deva", default=str(ROOT / "benchmark" / "hinglish_deva.tsv"),
                    help="Devanagari versions of Hinglish rows (id, text_deva); '' to disable")
    args = ap.parse_args()

    synth = Path(args.synth_dir)
    rows = list(csv.DictReader(open(synth / "manifest.csv", encoding="utf-8")))
    bench = {}
    if Path(args.benchmark).exists():
        bench = {r["id"]: r for r in csv.DictReader(open(args.benchmark, encoding="utf-8"), delimiter="\t")}

    deva = {}
    if args.deva and Path(args.deva).exists():
        deva = {r["id"]: r["text_deva"] for r in csv.DictReader(open(args.deva, encoding="utf-8"), delimiter="\t")}

    def scoring(row: dict) -> tuple[str, str, str]:
        """(whisper language key, reference text, metric name) for a row."""
        if row["lang"] == "hinglish" and row["id"] in deva:
            return "hi", deva[row["id"]], "hinglish_deva_rough"
        return row["lang"], row["text_in"], "standard"

    asr = load_whisper()
    hyps = transcribe(asr, [str(synth / f"{r['id']}.wav") for r in rows], [scoring(r)[0] for r in rows])

    out, by_group = [], defaultdict(list)
    for row, hyp in zip(rows, hyps):
        lang, ref_text, metric = scoring(row)
        ref = score_text(ref_text, lang)
        h = score_text(hyp, lang)
        category = bench.get(row["id"], {}).get("category", "fixture")
        result = {"id": row["id"], "lang": row["lang"], "category": category, "metric": metric,
                  "cer": round(cer(ref, h), 3), "wer": round(wer(ref, h), 3),
                  "ref_chars": len(ref), "hyp_chars": len(h), "reference": ref, "hypothesis": hyp}
        out.append(result)
        by_group[f"lang={row['lang']}"].append(result)
        by_group[f"category={category}"].append(result)
        print(f"{row['id']} {row['lang']:8} CER {result['cer']:.3f} WER {result['wer']:.3f} | {hyp}")

    with open(synth / "asr_eval.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    print()
    for key in sorted(by_group):
        g = by_group[key]
        print(f"{key:28} n={len(g):3}  CER {sum(r['cer'] for r in g) / len(g):.3f}  "
              f"WER {sum(r['wer'] for r in g) / len(g):.3f}")


if __name__ == "__main__":
    main()
