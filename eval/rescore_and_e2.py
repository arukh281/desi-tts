"""E0/E1 re-score + E2 benchmark in one Kaggle run, all with the same eval settings.

E0 and E1 audio comes from their earlier runs (attached kernel outputs) and is
only re-scored. E2 is trained on session 1 only (train/run_experiment.py
--sessions session_1), checkpoint picked on val, then benchmarked.

Usage (on Kaggle, KERNELS="desi-tts-e0-baseline desi-tts-e1-normalize",
DATASETS="desi-tts-reference desi-tts-own-voice"):
  python eval/rescore_and_e2.py --ref REF.wav ... -- <train_gpt.py options>
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IN = Path("/tmp/in")


def evaluate(*args: str) -> None:
    print("\n$ python eval/evaluate.py", " ".join(args), flush=True)
    subprocess.run([sys.executable, "eval/evaluate.py", *args], check=True, cwd=ROOT)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", nargs="+", required=True)
    ap.add_argument("train_args", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    evaluate("--name", "e0", "--ref", *args.ref, "--synth-from", str(IN / "desi-tts-e0-baseline/out/e0"))
    evaluate("--name", "e1", "--ref", *args.ref, "--synth-from", str(IN / "desi-tts-e1-normalize/out/e1"))
    subprocess.run([sys.executable, "train/run_experiment.py", "--name", "e2", "--sessions", "session_1",
                    "--data", str(IN / "desi-tts-own-voice"), "--ref", *args.ref, *args.train_args],
                   check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
