"""Build the Final system's per-seed results from the final run, no GPU.

Final = E3 + normaliser + Hinglish through the Devanagari route. The route only
changes Hinglish rows, so per seed: E3's rows for English and Hindi, and
E3_deva's rows for the 16 Hinglish sentences (same seeds, same run).

Usage:  python eval/build_final.py [--final results/final]
"""
import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final", type=Path, default=ROOT / "results" / "final")
    args = ap.parse_args()
    for seed in sorted(p.name for p in (args.final / "E3").glob("seed*")):
        out = args.final / "Final" / seed
        out.mkdir(parents=True, exist_ok=True)
        for name in ("asr_eval.csv", "manifest.csv"):
            e3 = list(csv.DictReader(open(args.final / "E3" / seed / name, encoding="utf-8")))
            deva = {r["id"]: r for r in csv.DictReader(open(args.final / "E3_deva" / seed / name, encoding="utf-8"))}
            rows = [deva.get(r["id"], r) if r["lang"] == "hinglish" else r for r in e3]
            fields = list(dict.fromkeys(k for r in rows for k in r))
            with open(out / name, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields, restval="")
                w.writeheader()
                w.writerows(rows)
        print(out)


if __name__ == "__main__":
    main()
