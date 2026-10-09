"""Timing sanity check on synthesised audio. No ML, runs anywhere.

Flags per clip:
  stretched          seconds per character > 1.5x the language median
  rushed             seconds per character < 0.6x the language median
  long_pause         a silence longer than 0.8 s inside the speech
  trailing_audio     the last 0.3 s is still loud (cut off mid-word, or babble at the end)
  hyp_longer         Whisper heard much more text than was asked for (needs asr_eval.csv)

Characters are counted on the spoken form that was synthesised (text_used).
Writes <synth-dir>/duration_check.csv.

Usage:  python eval/duration_check.py --synth-dir out/synth
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

STRETCH, RUSH = 1.5, 0.6
PAUSE_S = 0.8
TAIL_S = 0.3
TAIL_WITHIN_DB = 12.0  # tail this close to the speech level counts as "still talking"
HYP_LONGER = 1.5


def frame_db(audio: np.ndarray, sr: int, ms: int = 20) -> np.ndarray:
    n = int(sr * ms / 1000)
    frames = audio[: len(audio) // n * n].reshape(-1, n)
    return 20 * np.log10(np.sqrt((frames**2).mean(axis=1)) + 1e-9)


def timing_flags(audio: np.ndarray, sr: int) -> list[str]:
    db = frame_db(audio, sr)
    if db.size == 0:
        return ["empty"]
    speech = np.percentile(db, 90)
    active = db > speech - 30
    idx = np.flatnonzero(active)
    flags = []
    if idx.size:
        inner = ~active[idx[0]:idx[-1] + 1]
        longest, run = 0, 0
        for silent in inner:
            run = run + 1 if silent else 0
            longest = max(longest, run)
        if longest * 0.02 > PAUSE_S:
            flags.append(f"long_pause {longest * 0.02:.1f}s")
    tail = db[-int(TAIL_S / 0.02):]
    if tail.size and tail.mean() > speech - TAIL_WITHIN_DB:
        flags.append("trailing_audio")
    return flags


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synth-dir", required=True)
    synth = Path(ap.parse_args().synth_dir)

    rows = [r for r in csv.DictReader(open(synth / "manifest.csv", encoding="utf-8")) if not r.get("error")]
    asr = {}
    if (synth / "asr_eval.csv").exists():
        asr = {r["id"]: r for r in csv.DictReader(open(synth / "asr_eval.csv", encoding="utf-8"))}

    per_char = {r["id"]: float(r["audio_s"]) / max(1, len(r["text_used"])) for r in rows}
    by_lang = defaultdict(list)
    for r in rows:
        by_lang[r["lang"]].append(per_char[r["id"]])
    median = {lang: float(np.median(v)) for lang, v in by_lang.items()}
    print("median s/char: " + ", ".join(f"{lang} {m:.3f}" for lang, m in sorted(median.items())))

    out = []
    for r in rows:
        ratio = per_char[r["id"]] / median[r["lang"]]
        audio, sr = sf.read(synth / f"{r['id']}.wav", dtype="float32")
        flags = timing_flags(audio, sr)
        if ratio > STRETCH:
            flags.append(f"stretched {ratio:.2f}x")
        elif ratio < RUSH:
            flags.append(f"rushed {ratio:.2f}x")
        a = asr.get(r["id"])
        if a and int(a["hyp_chars"]) > HYP_LONGER * int(a["ref_chars"]):
            flags.append(f"hyp_longer {a['hyp_chars']}/{a['ref_chars']} chars")
        out.append({"id": r["id"], "lang": r["lang"], "s_per_char": round(per_char[r["id"]], 4),
                    "vs_median": round(ratio, 2), "flags": "; ".join(flags)})
        print(f"{r['id']} {r['lang']:8} {per_char[r['id']]:.3f} s/char ({ratio:.2f}x)  {'; '.join(flags)}")

    with open(synth / "duration_check.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)


if __name__ == "__main__":
    main()
