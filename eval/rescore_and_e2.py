"""E0/E1 re-score + E2 benchmark in one Kaggle run, all with the same eval settings.

E0 and E1 audio comes from their earlier runs (attached kernel outputs) and is
only re-scored. E2 is synthesised from its already-selected checkpoint (raw
text and normalised text), so it doesn't need retraining.

Usage (on Kaggle, KERNELS="desi-tts-e0-baseline desi-tts-e1-normalize desi-tts-e2-finetune-s1"):
  python eval/rescore_and_e2.py --ref REF.wav ...
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
    args = ap.parse_args()

    evaluate("--name", "e0", "--ref", *args.ref, "--synth-from", str(IN / "desi-tts-e0-baseline/out/e0"))
    evaluate("--name", "e1", "--ref", *args.ref, "--synth-from", str(IN / "desi-tts-e1-normalize/out/e1"))
    model = str(IN / "desi-tts-e2-finetune-s1/out/e2/select/export_250")
    evaluate("--name", "e2_raw", "--model", model, "--ref", *args.ref, "--normalize", "off", "--stream")
    evaluate("--name", "e2_norm", "--model", model, "--ref", *args.ref, "--normalize", "on", "--stream")


if __name__ == "__main__":
    main()
