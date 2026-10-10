"""E4: E3's recipe, but Hinglish transcripts in Devanagari with language "hi".

Trains with train/run_experiment.py --hinglish-as-hindi (same audio, same
seed and steps; checkpoint picked on val by the usual rule), keeps the weights,
then synthesises the 10 held-out Hinglish sentences with 3 seeds (Hinglish
sent as Devanagari/hi, as it was trained). Compare with E3 + inference
transliteration on the same sentences (E3_heldout_deva in eval/final_seeds.py).

Usage (on Kaggle):  python eval/block11_e4.py --data <dataset> --ref REF.wav ... -- <train_gpt.py options>
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
    subprocess.run([sys.executable, "train/run_experiment.py", "--name", "e4", "--data", args.data, "--ref", *args.ref,
                    "--hinglish-as-hindi", "--bench", "", "--keep-model", *args.train_args], check=True, cwd=ROOT)
    chosen = json.load(open(out / "e4" / "select" / "selected.json"))["export"]
    heldout = str(ROOT / "benchmark" / "hinglish_heldout.tsv")
    for seed in ("1", "2", "3"):
        env = dict(os.environ, OUT_DIR=str(out / "e4" / "heldout"))
        subprocess.run([sys.executable, "eval/evaluate.py", "--name", f"seed{seed}", "--model", chosen, "--ref", *args.ref,
                        "--benchmark", heldout, "--normalize", "on", "--hinglish-route", "deva", "--seed", seed],
                       check=True, cwd=ROOT, env=env)


if __name__ == "__main__":
    main()
