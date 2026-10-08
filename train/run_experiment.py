"""One fine-tuning experiment end to end, in one Kaggle run.

  1. train/train_gpt.py on the data (milestones at 25/50/75/100%)
  2. train/select_checkpoint.py picks one using the val split only
  3. eval/evaluate.py runs the benchmark with the chosen checkpoint, raw text
     and normalised text, same reference clips and seed as the baselines

Usage (on Kaggle):
  python train/run_experiment.py --name e2 --data /tmp/in/desi-tts-own-voice \\
      --ref /tmp/in/desi-tts-reference/ref_en.wav ... [train_gpt.py options after --]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> None:
    print("\n$ python", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], check=True, cwd=ROOT)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--ref", nargs="+", required=True)
    ap.add_argument("train_args", nargs=argparse.REMAINDER, help="passed to train_gpt.py (after --)")
    args = ap.parse_args()
    extra = [a for a in args.train_args if a != "--"]

    out = Path(os.environ.get("OUT_DIR", "outputs")) / args.name
    train, select = out / "train", out / "select"
    run("train/train_gpt.py", "--data", args.data, "--out", str(train), *extra)
    run("train/select_checkpoint.py", "--run", str(train), "--data", args.data, "--ref", *args.ref,
        "--out", str(select))
    chosen = json.load(open(select / "selected.json"))["export"]
    for norm in ("off", "on"):
        os.environ["OUT_DIR"] = str(out)
        run("eval/evaluate.py", "--name", f"bench_norm_{norm}", "--model", chosen, "--ref", *args.ref,
            "--normalize", norm, "--stream")

    # Keep logs, CSVs and audio; drop the multi-GB weights so Kaggle doesn't ship them back.
    for p in list(out.rglob("*.pth")):
        p.unlink()
    shutil.rmtree(train / "milestones", ignore_errors=True)
    print("done:", out)


if __name__ == "__main__":
    main()
