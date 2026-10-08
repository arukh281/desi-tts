"""P1 probe: how does the pretrained model do with my voice as the reference?

Synthesises 3 fixture sentences (en f01, Hinglish f03, Hindi f05) with --ref on
my reference clips, once per seed (1, 2, 3), plus the Hinglish sentence written
in Devanagari and sent as language "hi". The question there: does XTTS read
Hinglish better through its Hindi path than as Roman text through "en"?
Each seed folder then gets the Whisper check (eval/asr_eval.py).

Usage (on Kaggle, with the reference dataset attached):
  python infer/probe.py --ref /kaggle/input/desi-tts-reference/ref_en.wav ...
"""
import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_IDS = ["f01", "f03", "f05"]
# f03 "Aapka refund teen din mein account mein aa jayega." written in Devanagari.
F03_DEVANAGARI = "आपका रिफंड तीन दिन में अकाउंट में आ जाएगा।"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", default=["1", "2", "3"])
    args = ap.parse_args()

    out = Path(os.environ.get("OUT_DIR", "outputs")) / "probe"
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in csv.DictReader(open(ROOT / "infer" / "fixture.tsv", encoding="utf-8"), delimiter="\t")
            if r["id"] in FIXTURE_IDS]
    rows.append({"id": "f03_dev", "lang": "hi", "text": F03_DEVANAGARI})
    probe_tsv = out / "probe.tsv"
    with open(probe_tsv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "lang", "text"], delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    for seed in args.seeds:
        folder = out / f"seed{seed}"
        subprocess.run([sys.executable, "infer/synthesize.py", "--model", "pretrained", "--ref", *args.ref,
                        "--benchmark", str(probe_tsv), "--seed", seed, "--out", str(folder)], check=True, cwd=ROOT)
        subprocess.run([sys.executable, "eval/asr_eval.py", "--synth-dir", str(folder),
                        "--benchmark", str(probe_tsv)], check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
