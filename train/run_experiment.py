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


def subset(data: Path, sessions: list[str], dest: Path) -> str:
    """Metadata filtered to some recording sessions (wavs linked, not copied)."""
    import csv

    keep = {r["id"] for r in csv.DictReader(open(data / "keep_drop.csv", encoding="utf-8"))
            if r["session"] in sessions}
    dest.mkdir(parents=True, exist_ok=True)
    if not (dest / "wavs").exists():
        (dest / "wavs").symlink_to(data / "wavs")
    counts = {}
    for meta in data.glob("metadata_*.csv"):
        lines = meta.read_text(encoding="utf-8").splitlines()
        rows = [l for l in lines[1:] if Path(l.split("|")[0]).stem in keep]
        (dest / meta.name).write_text("\n".join([lines[0], *rows]) + "\n", encoding="utf-8")
        counts[meta.name] = len(rows)
    print(f"training subset {sessions}: {counts}")
    return str(dest)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--ref", nargs="+", required=True)
    ap.add_argument("--sessions", help="comma-separated sessions to train on, e.g. session_1 (default: all)")
    ap.add_argument("--bench", default="raw,norm",
                    help="benchmark variants: raw (no normaliser), norm (normaliser), deva (normaliser + "
                         "Hinglish via Devanagari/hi)")
    ap.add_argument("train_args", nargs=argparse.REMAINDER, help="passed to train_gpt.py (after --)")
    args = ap.parse_args()
    variants = {"raw": ["--normalize", "off"], "norm": ["--normalize", "on"],
                "deva": ["--normalize", "on", "--hinglish-route", "deva"]}
    names = {"raw": "bench_norm_off", "norm": "bench_norm_on", "deva": "bench_deva"}
    args.bench = [(names[v], variants[v]) for v in args.bench.split(",")]
    extra = [a for a in args.train_args if a != "--"]

    out = Path(os.environ.get("OUT_DIR", "outputs")) / args.name
    train, select = out / "train", out / "select"
    if args.sessions:
        args.data = subset(Path(args.data), args.sessions.split(","), out / "data_subset")
    run("train/train_gpt.py", "--data", args.data, "--out", str(train), *extra)
    run("train/select_checkpoint.py", "--run", str(train), "--data", args.data, "--ref", *args.ref,
        "--out", str(select))
    chosen = json.load(open(select / "selected.json"))["export"]
    os.environ["OUT_DIR"] = str(out)
    for name, extra_eval in args.bench:
        run("eval/evaluate.py", "--name", name, "--model", chosen, "--ref", *args.ref, "--stream", *extra_eval)

    # Keep logs, CSVs and audio; drop the multi-GB weights so Kaggle doesn't ship them back.
    for p in list(out.rglob("*.pth")):
        p.unlink()
    shutil.rmtree(train / "milestones", ignore_errors=True)
    print("done:", out)


if __name__ == "__main__":
    main()
