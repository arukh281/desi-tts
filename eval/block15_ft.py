"""Block 15 bonus: E3's recipe with name clips upweighted, scored on the held-out names.

Trains with train/run_experiment.py (sessions 1+2, lr 5e-6, batch 2, accum 8,
2000 steps, seed 1234, same val-only checkpoint rule) plus --upweight-names 2,
then synthesises benchmark/names_heldout.tsv with seeds 1-3 (final pipeline
settings) for comparison with Block 15's "base" (the final model). The
benchmark is not touched. Weights are not kept unless it wins.

Usage (on Kaggle, DATASETS="desi-tts-own-voice desi-tts-reference"):
  python eval/block15_ft.py --data /tmp/in/desi-tts-own-voice --ref REF.wav ...
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--ref", nargs="+", required=True)
    args = ap.parse_args()
    out = Path(os.environ.get("OUT_DIR", "outputs"))
    subprocess.run([sys.executable, "train/run_experiment.py", "--name", "e3names", "--data", args.data,
                    "--ref", *args.ref, "--bench", "", "--keep-model", "--upweight-names", "2", "--",
                    "--lr", "5e-6", "--batch", "2", "--grad-accum", "8", "--max-steps", "2000",
                    "--save-step", "2000", "--keep", "1", "--eval-step", "200", "--seed", "1234"],
                   check=True, cwd=ROOT)
    chosen = json.load(open(out / "e3names" / "select" / "selected.json"))["export"]
    subprocess.run([sys.executable, "eval/block15.py", "--stage", "heldout", "--only", "base",
                    "--model", chosen, "--ref", *args.ref], check=True, cwd=ROOT,
                   env=dict(os.environ, OUT_DIR=str(out / "e3names")))
    # Drop the weights; this run is only a comparison.
    for p in (out / "e3names").rglob("*.pth"):
        p.unlink()


if __name__ == "__main__":
    main()
