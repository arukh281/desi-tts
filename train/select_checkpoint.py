"""Pick a checkpoint using the val split only. The benchmark is never used here.

For each milestone saved by train_gpt.py (25/50/75/100% of training) it:
  1. exports it (train/export.py)
  2. synthesises the val clips' texts with my reference clips
  3. scores them: Whisper CER on English and Hindi val clips (Hinglish CER is
     meaningless, see eval/asr_eval.py), duration flags, and the val loss
     logged nearest to that step

Rule: lowest val loss, among milestones whose val CER (en + hi) is within 0.05
of the best and whose duration flags are within 2 of the fewest. Val loss is
the main signal because CER on ~10 val clips is too noisy to rank by: on E2 it
picked step 250 of 1000 and the voice came out barely adapted. The ASR and
duration checks stay as guards against a checkpoint that babbles.
Writes selection.csv (all milestones) and selected.json.

Usage (on Kaggle):
  python train/select_checkpoint.py --run OUT/train --data DATA --ref REF.wav [...]
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "recording"))
from order import language_group  # noqa: E402


def run(*args: str) -> None:
    print("$ python", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], check=True, cwd=ROOT)


def val_set(data: Path, out: Path) -> Path:
    """The val clips' texts as a benchmark-style TSV (id, lang, text)."""
    rows = []
    for lang in ("en", "hi"):
        path = data / f"metadata_{lang}_val.csv"
        if path.exists():
            for r in csv.DictReader(open(path, encoding="utf-8"), delimiter="|"):
                row = {"id": Path(r["audio_file"]).stem, "lang": lang, "text": r["text"]}
                row["lang"] = language_group(row)
                rows.append(row)
    tsv = out / "val.tsv"
    with open(tsv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "lang", "text"], delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    return tsv


def val_loss_near(losses: Path, step: int) -> float | None:
    vals = [(int(r["step"]), float(r["loss"])) for r in csv.DictReader(open(losses))
            if r["split"] == "val" and r["loss"]]
    if not vals:
        return None
    return min(vals, key=lambda v: abs(v[0] - step))[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="train_gpt.py --out folder")
    ap.add_argument("--data", required=True, help="training data folder (for metadata_*_val.csv)")
    ap.add_argument("--ref", nargs="+", required=True)
    ap.add_argument("--out", default=os.path.join(os.environ.get("OUT_DIR", "outputs"), "select"))
    args = ap.parse_args()

    run_dir, out = Path(args.run), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tsv = val_set(Path(args.data), out)
    results = []
    for ckpt in sorted((run_dir / "milestones").glob("step_*.pth"), key=lambda p: int(p.stem.split("_")[1])):
        step = int(ckpt.stem.split("_")[1])
        export, synth = out / f"export_{step}", out / f"val_{step}"
        run("train/export.py", "--run", str(run_dir), "--checkpoint", str(ckpt), "--out", str(export))
        run("infer/synthesize.py", "--model", str(export), "--ref", *args.ref, "--benchmark", str(tsv),
            "--out", str(synth))
        run("eval/asr_eval.py", "--synth-dir", str(synth), "--benchmark", str(tsv), "--deva", "")
        run("eval/duration_check.py", "--synth-dir", str(synth))
        asr = [r for r in csv.DictReader(open(synth / "asr_eval.csv", encoding="utf-8")) if r["lang"] != "hinglish"]
        flags = sum(1 for r in csv.DictReader(open(synth / "duration_check.csv")) if r["flags"])
        results.append({"step": step, "val_cer_en_hi": round(sum(float(r["cer"]) for r in asr) / len(asr), 4),
                        "duration_flags": flags, "val_loss": val_loss_near(run_dir / "losses.csv", step),
                        "export": str(export)})
        print(results[-1], flush=True)
        (export / "model.pth").unlink()  # keep disk free; re-export the chosen one below

    best_cer = min(r["val_cer_en_hi"] for r in results)
    fewest_flags = min(r["duration_flags"] for r in results)
    ok = [r for r in results if r["val_cer_en_hi"] <= best_cer + 0.05 and r["duration_flags"] <= fewest_flags + 2]
    chosen = min(ok, key=lambda r: r["val_loss"] if r["val_loss"] is not None else float("inf"))
    with open(out / "selection.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    run("train/export.py", "--run", str(run_dir), "--checkpoint",
        str(run_dir / "milestones" / f"step_{chosen['step']}.pth"), "--out", chosen["export"])
    json.dump(chosen, open(out / "selected.json", "w"), indent=2)
    print("selected:", chosen)


if __name__ == "__main__":
    main()
