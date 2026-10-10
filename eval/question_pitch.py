"""Do yes/no questions end with a rise? Pitch slope over the last ~300 ms. CPU only.

For each yes/no question in the benchmark, trailing silence is trimmed, pitch
is tracked with pyin (65-400 Hz), and a line is fitted to the voiced frames of
the final 300 ms, in semitones per second. "Rises" = slope above 0.

Final = E3 audio for English and Hindi rows and the Devanagari-route audio for
Hinglish rows (the same files as the sample page and listening test).

Usage:
  python eval/question_pitch.py --e1 E1_WAV_DIR --final E3_WAV_DIR --final-hinglish E3_DEVA_WAV_DIR
Writes results/question_pitch.md.
"""
import argparse
import statistics
from pathlib import Path

import librosa
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Yes/no form (b057 "How would you rate..." is a wh-question, so it's left out).
YES_NO = ["b003", "b009", "b029", "b053", "b054", "b055", "b056", "b064", "b075", "b080"]
TAIL_S = 0.3


def end_slope(path: Path) -> float | None:
    y, sr = librosa.load(path, sr=16000)
    y, _ = librosa.effects.trim(y, top_db=35)
    tail = y[-int(0.6 * sr):]  # a little extra so pyin has context
    f0, voiced, _ = librosa.pyin(tail, fmin=65, fmax=400, sr=sr, frame_length=1024, hop_length=160)
    t = librosa.times_like(f0, sr=sr, hop_length=160)
    keep = voiced & (t >= t[-1] - TAIL_S) & ~np.isnan(f0)
    if keep.sum() < 5:
        return None
    semitones = 12 * np.log2(f0[keep])
    return float(np.polyfit(t[keep], semitones, 1)[0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--e1", type=Path, required=True)
    ap.add_argument("--final", type=Path, required=True)
    ap.add_argument("--final-hinglish", type=Path, required=True)
    args = ap.parse_args()
    hinglish = {"b064", "b075"}
    rows, slopes = [], {"E1": [], "Final": []}
    for i in YES_NO:
        e1 = end_slope(args.e1 / f"{i}.wav")
        fin = end_slope((args.final_hinglish if i in hinglish else args.final) / f"{i}.wav")
        for name, v in (("E1", e1), ("Final", fin)):
            if v is not None:
                slopes[name].append(v)
        fmt = lambda v: "unvoiced" if v is None else f"{v:+.1f}"
        rows.append(f"| {i} | {fmt(e1)} | {fmt(fin)} |")
    lines = ["<!-- python eval/question_pitch.py; seed 1234 audio (E1 single run, E3 run 1); slope in semitones/s over the last 300 ms -->",
             "| row | E1 slope | Final slope |", "|---|---|---|", *rows, ""]
    for name, v in slopes.items():
        lines.append(f"- {name}: rises on {sum(s > 0 for s in v)}/{len(v)}, median slope {statistics.median(v):+.1f} st/s")
    text = "\n".join(lines) + "\n"
    (ROOT / "results" / "question_pitch.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
