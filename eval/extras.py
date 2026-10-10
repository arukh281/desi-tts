"""Block 10 extras for one fine-tuned model, in one Kaggle run.

  2x2      {pretrained, fine-tuned} x {Roman/en, IndicXlit Devanagari/hi}
           on the 16 Hinglish benchmark rows
  heldout  the same 2x2 on 10 new Hinglish sentences that are in neither the
           benchmark nor the prompts (benchmark/hinglish_heldout.tsv)
  ceiling  speaker similarity of my REAL val recordings vs the reference clips
           (same encoder), i.e. what a perfect copy of my voice would score
  profile  infer/profile.py on the fine-tuned model (normaliser + IndicXlit)

Usage (on Kaggle):
  python eval/extras.py --model <export dir> --data <processed dataset> --ref REF.wav ...
"""
import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "recording"))
from order import language_group  # noqa: E402

OUT = Path(os.environ.get("OUT_DIR", "outputs")) / "extras"


def py(*args: str) -> None:
    print("\n$ python", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], check=True, cwd=ROOT)


def hinglish_rows() -> Path:
    rows = [r for r in csv.DictReader(open(ROOT / "benchmark" / "benchmark.tsv", encoding="utf-8"), delimiter="\t")
            if r["lang"] == "hinglish"]
    path = OUT / "hinglish16.tsv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "lang", "category", "text"], delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


def two_by_two(name: str, texts: Path, deva: Path, model: str, ref: list[str]) -> None:
    for model_name, model_path in (("pretrained", "pretrained"), ("finetuned", model)):
        for route in ("en", "deva"):
            synth = OUT / name / f"{model_name}_{route}"
            py("infer/synthesize.py", "--model", model_path, "--ref", *ref, "--benchmark", str(texts),
               "--normalize", "on", "--hinglish-route", route, "--out", str(synth))
            py("eval/asr_eval.py", "--synth-dir", str(synth), "--benchmark", str(texts), "--deva", str(deva))
            py("eval/speaker_sim.py", "--synth-dir", str(synth), "--ref", *ref)


def ceiling(data: Path, ref: list[str]) -> None:
    """Real recordings in the 'synth dir' layout so speaker_sim.py can score them."""
    folder = OUT / "ceiling_real_val"
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for lang in ("en", "hi"):
        meta = data / f"metadata_{lang}_val.csv"
        if not meta.exists():
            continue
        for r in csv.DictReader(open(meta, encoding="utf-8"), delimiter="|"):
            clip = Path(r["audio_file"]).stem
            row = {"id": clip, "lang": lang, "text": r["text"]}
            row["lang"] = language_group(row)
            link = folder / f"{clip}.wav"
            if not link.exists():
                link.symlink_to(data / r["audio_file"])
            rows.append({"id": clip, "lang": row["lang"]})
    with open(folder / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "lang"])
        w.writeheader()
        w.writerows(rows)
    py("eval/speaker_sim.py", "--synth-dir", str(folder), "--ref", *ref)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--ref", nargs="+", required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    ceiling(Path(args.data), args.ref)
    two_by_two("hinglish16", hinglish_rows(), ROOT / "benchmark" / "hinglish_deva.tsv", args.model, args.ref)
    heldout = ROOT / "benchmark" / "hinglish_heldout.tsv"
    two_by_two("heldout10", heldout, heldout, args.model, args.ref)
    os.environ["OUT_DIR"] = str(OUT)
    py("infer/profile.py", "--model", args.model, "--ref", *args.ref, "--repeat", "2")


if __name__ == "__main__":
    main()
