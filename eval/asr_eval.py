"""Intelligibility: transcribe synthesised audio with Whisper and score CER/WER.

Reference = the expected spoken form of the input text (text/normalize.py,
which reproduces the benchmark's check column). Hypothesis = Whisper's
transcript, normalised the same way. Language forced per row (see asr_common.py).

Input is a folder from infer/synthesize.py (manifest.csv + <id>.wav).
Writes <synth-dir>/asr_eval.csv (per row) and prints mean CER/WER per category
and language. Category comes from --benchmark when the id is in it.

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
    args = ap.parse_args()

    synth = Path(args.synth_dir)
    rows = list(csv.DictReader(open(synth / "manifest.csv", encoding="utf-8")))
    bench = {}
    if Path(args.benchmark).exists():
        bench = {r["id"]: r for r in csv.DictReader(open(args.benchmark, encoding="utf-8"), delimiter="\t")}

    asr = load_whisper()
    hyps = transcribe(asr, [str(synth / f"{r['id']}.wav") for r in rows], [r["lang"] for r in rows])

    out, by_group = [], defaultdict(list)
    for row, hyp in zip(rows, hyps):
        ref = score_text(row["text_in"], row["lang"])
        h = score_text(hyp, row["lang"])
        category = bench.get(row["id"], {}).get("category", "fixture")
        result = {"id": row["id"], "lang": row["lang"], "category": category,
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
