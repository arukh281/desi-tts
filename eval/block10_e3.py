"""Retrain E3 with its exact recipe (seed 1234), keep the chosen checkpoint, run eval/extras.py on it.

E3's weights weren't kept the first time. Training is seeded, so this should
pick the same checkpoint; the selection table is written next to the first
run's for comparison.

Usage (on Kaggle):  python eval/block10_e3.py --data <dataset> --ref REF.wav ... -- <train_gpt.py options>
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
    ap.add_argument("train_args", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    out = Path(os.environ.get("OUT_DIR", "outputs"))
    subprocess.run([sys.executable, "train/run_experiment.py", "--name", "e3", "--data", args.data,
                    "--ref", *args.ref, "--bench", "", "--keep-model", *args.train_args], check=True, cwd=ROOT)
    chosen = json.load(open(out / "e3" / "select" / "selected.json"))
    print("chosen:", chosen, flush=True)
    subprocess.run([sys.executable, "eval/extras.py", "--model", chosen["export"], "--data", args.data,
                    "--ref", *args.ref], check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
