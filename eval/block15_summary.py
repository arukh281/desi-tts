"""Summarise Block 15 runs: CER, name words missed, babble flags and retries per setting, plus a
paired bootstrap of each setting against a baseline (3 seeds, crashed rows = CER 1.0).

Usage:  python eval/block15_summary.py --dir RUN_DIR [--base base] [--text benchmark/names_heldout.tsv] --out results/FILE.md
RUN_DIR holds <setting>/seed*/ folders.
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--base", default="base")
    ap.add_argument("--text", default=str(ROOT / "benchmark" / "names_heldout.tsv"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    settings = sorted(p.name for p in args.dir.iterdir() if p.is_dir())
    lines = [f"<!-- python eval/block15_summary.py --dir {args.dir.name} -->", "",
             "| setting | mean CER | name words missed | duration flags | guard retries |", "|---|---|---|---|---|"]
    for s in settings:
        seeds = sorted((args.dir / s).glob("seed*"))
        cer = [float(r["cer"]) for d in seeds for r in csv.DictReader(open(d / "asr_eval.csv", encoding="utf-8"))]
        flags = sum(bool(r["flags"]) for d in seeds for r in csv.DictReader(open(d / "duration_check.csv", encoding="utf-8")))
        retries = sum(int(r.get("retries") or 0) for d in seeds for r in csv.DictReader(open(d / "manifest.csv", encoding="utf-8")))
        names = subprocess.run([sys.executable, str(ROOT / "eval" / "names_seen.py"), f"--benchmark={args.text}",
                                f"{s}=" + ",".join(str(d) for d in seeds)], capture_output=True, text=True).stdout
        missed = total = 0
        for line in names.splitlines():
            for cell in line.split("|")[3:5]:
                if "/" in cell:
                    m, t = cell.split("=")[0].strip().split("/")
                    missed, total = missed + int(m), total + int(t)
        lines.append(f"| {s} | {sum(cer) / len(cer):.3f} | {missed}/{total} = {missed / total:.2f} | {flags} | {retries} |")
    lines.append("")
    for s in settings:
        if s != args.base:
            out = subprocess.run([sys.executable, str(ROOT / "eval" / "bootstrap.py"), "--a", f"{args.base}={args.dir / args.base}",
                                  "--b", f"{s}={args.dir / s}"], capture_output=True, text=True).stdout
            lines += [l for l in out.splitlines() if l.startswith("| all") or l.startswith("### CER:")] + [""]
    text = "\n".join(lines) + "\n"
    args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
