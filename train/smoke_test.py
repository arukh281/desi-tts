"""End-to-end training smoke test on Kaggle, on stock-voice data (no real result).

  1. make ~40 stock-voice clips (train/make_smoke_data.py)
  2. Whisper transcript check on them (data_prep/asr_check.py), which must catch smoke_wrong
  3. train run A to --steps-a, saving a checkpoint
  4. train run B resuming from A to --steps-b: the step count must carry on
  5. export the last checkpoint and synthesise with it (infer/synthesize.py)
Then it checks: losses finite and falling, step count kept across the resume.
Checkpoints are deleted at the end so Kaggle doesn't ship GBs back.

Usage (on Kaggle):  python train/smoke_test.py [--steps-a 40 --steps-b 80]
"""
import argparse
import csv
import json
import math
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
    ap.add_argument("--steps-a", type=int, default=40)
    ap.add_argument("--steps-b", type=int, default=80)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    out = Path(os.environ.get("OUT_DIR", "outputs")) / "smoke"
    data, a, b, export = out / "data", out / "run_a", out / "run_b", out / "export"
    common = ["--data", str(data), "--batch", str(args.batch), "--grad-accum", "1",
              "--save-step", "20", "--eval-step", "20", "--keep", "1"] + (["--fp16"] if args.fp16 else [])

    run("train/make_smoke_data.py", "--out", str(data))
    run("data_prep/asr_check.py", "--processed", str(data), "--out", str(out / "asr"))
    run("train/train_gpt.py", "--out", str(a), "--max-steps", str(args.steps_a), *common)
    run("train/train_gpt.py", "--out", str(b), "--max-steps", str(args.steps_b), "--resume", str(a), *common)
    run("train/export.py", "--run", str(b), "--out", str(export))
    run("infer/synthesize.py", "--model", str(export), "--speaker", "Ana Florence",
        "--benchmark", "infer/fixture.tsv", "--out", str(out / "synth_after"))

    checks = {}
    asr = {r["id"]: float(r["cer"]) for r in csv.DictReader(open(out / "asr" / "asr.csv", encoding="utf-8"))}
    checks["asr_catches_wrong_transcript"] = asr["smoke_wrong"] > 0.3
    checks["asr_cer_median_others"] = sorted(v for k, v in asr.items() if k != "smoke_wrong")[len(asr) // 2]
    sa, sb = (json.load(open(d / "train_summary.json")) for d in (a, b))
    checks["steps_run_a"], checks["steps_run_b"] = sa["steps_done"], sb["steps_done"]
    checks["resume_kept_step_count"] = sb["steps_done"] > sa["steps_done"]
    losses = [float(r["loss"]) for d in (a, b) for r in csv.DictReader(open(d / "losses.csv"))
              if r["split"] == "train" and r["loss"]]
    checks["losses_finite"] = bool(losses) and all(math.isfinite(x) for x in losses)
    k = max(1, len(losses) // 5)
    checks["loss_first_fifth"] = round(sum(losses[:k]) / k, 4)
    checks["loss_last_fifth"] = round(sum(losses[-k:]) / k, 4)
    checks["loss_falling"] = checks["loss_last_fifth"] < checks["loss_first_fifth"]
    sec = [float(r["sec_per_step"]) for d in (a, b) for r in csv.DictReader(open(d / "losses.csv"))
           if r["sec_per_step"]]
    checks["median_sec_per_step"] = sorted(sec)[len(sec) // 2] if sec else None
    checks["peak_vram_gb_train"] = max(sa["peak_vram_gb"], sb["peak_vram_gb"])
    checks["trainable_params"], checks["total_params"] = sa["trainable_params"], sa["total_params"]
    checks["checkpoint_loads_in_synthesize"] = (out / "synth_after" / "manifest.csv").exists()
    json.dump(checks, open(out / "smoke_checks.json", "w"), indent=2)
    print(json.dumps(checks, indent=2))

    for p in list(out.rglob("*.pth")):
        p.unlink()
    shutil.rmtree(data / "wavs", ignore_errors=True)
    passed = all(v for k, v in checks.items() if isinstance(v, bool))
    print("SMOKE TEST", "PASSED" if passed else "FAILED")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
