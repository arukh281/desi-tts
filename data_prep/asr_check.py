"""Step 2 of data prep (needs a GPU, run on Kaggle): does each clip say its prompt?

Transcribes every processed clip with Whisper (eval/asr_common.py: large-v3,
language forced per row) and writes asr.csv with the CER between the prompt
text and the transcript, both normalised the same way. finalize.py turns CER
into keep / flag / drop using the thresholds in data_prep/config.json.

Usage (on Kaggle, with data/processed attached as a private dataset):
  python data_prep/asr_check.py --processed /kaggle/input/desi-tts-own-voice
Output: $OUT_DIR/asr.csv (copy it into data/processed/ before finalize.py)
"""
import argparse
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "recording"))
from asr_common import cer, load_whisper, score_text, transcribe  # noqa: E402
from order import language_group  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed", required=True, help="folder with clips.csv and wavs/")
    ap.add_argument("--out", default=os.environ.get("OUT_DIR", "."))
    args = ap.parse_args()

    processed = Path(args.processed)
    clips = [c for c in csv.DictReader(open(processed / "clips.csv", encoding="utf-8"))
             if (processed / "wavs" / f"{c['id']}.wav").exists()]
    groups = [language_group(c) for c in clips]
    hyps = transcribe(load_whisper(), [str(processed / "wavs" / f"{c['id']}.wav") for c in clips], groups)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "asr.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "group", "hypothesis", "cer"])
        for clip, group, hyp in zip(clips, groups, hyps):
            score = cer(score_text(clip["text"], group), score_text(hyp, group))
            writer.writerow([clip["id"], group, hyp, round(score, 3)])
            print(f"{clip['id']} {group:8} CER {score:.3f} | {hyp}")
    print(f"wrote {len(clips)} rows to {out / 'asr.csv'}")


if __name__ == "__main__":
    main()
