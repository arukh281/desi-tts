"""Build a small fake recording session where every clip has one known problem.

Used to test the data-prep checks without real recordings. The "speech" is a
harmonic tone with a syllable-like envelope, which is enough for level, SNR,
trimming and click detection (not for Whisper).

  syn_clean   nothing wrong
  syn_clip    driven into clipping
  syn_pause   2.5 s gap in the middle
  syn_click   Enter-key clicks at both edges
  syn_noisy   loud background noise
  syn_long    13 s, over the 11.6 s limit
  syn_quiet   peak around -40 dBFS
  syn_wrong   clean audio, but the transcript check will be fed a wrong text

Usage:  python data_prep/make_synthetic.py --out /tmp/syn
Writes <out>/raw/session_1/*.wav and <out>/prompts.tsv.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 48_000
RNG = np.random.default_rng(0)


def speech(seconds: float, level: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    tone = sum(np.sin(2 * np.pi * 140 * k * t) / k for k in range(1, 8))
    envelope = 0.55 + 0.45 * np.sin(2 * np.pi * 4 * t)  # ~4 syllables a second
    out = tone * envelope
    return (level * out / np.abs(out).max()).astype("float32")


def room(seconds: float, level_db: float = -65.0) -> np.ndarray:
    return (10 ** (level_db / 20) * RNG.standard_normal(int(seconds * SR))).astype("float32")


def with_room(*parts: np.ndarray) -> np.ndarray:
    audio = np.concatenate(parts)
    return audio + room(len(audio) / SR)


def click() -> np.ndarray:
    c = np.zeros(int(0.01 * SR), dtype="float32")
    c[:200] = 0.6 * np.exp(-np.arange(200) / 30) * np.sign(RNG.standard_normal(200))
    return c


CLIPS = {
    "syn_clean": lambda: with_room(room(0.4), speech(3), room(0.4)),
    "syn_clip": lambda: np.clip(with_room(room(0.4), speech(3, level=1.6), room(0.4)), -1, 1),
    "syn_pause": lambda: with_room(room(0.4), speech(1.5), np.zeros(int(2.5 * SR), "float32"), speech(1.5), room(0.4)),
    "syn_click": lambda: with_room(room(0.04), click(), room(0.3), speech(3), room(0.3), click(), room(0.04)),
    "syn_noisy": lambda: with_room(room(0.4), speech(3), room(0.4)) + room(3.8, level_db=-25),
    "syn_long": lambda: with_room(room(0.4), speech(13), room(0.4)),
    "syn_quiet": lambda: with_room(room(0.4), speech(3, level=0.01), room(0.4)),
    "syn_wrong": lambda: with_room(room(0.4), speech(3), room(0.4)),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    out = Path(ap.parse_args().out)
    session = out / "raw" / "session_1"
    session.mkdir(parents=True, exist_ok=True)
    for name, make in CLIPS.items():
        sf.write(session / f"{name}.wav", make(), SR, subtype="PCM_24")
    with open(out / "prompts.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["id", "lang", "text"])
        for i, name in enumerate(CLIPS):
            w.writerow([name, "hi" if i % 3 == 2 else "en", f"synthetic sentence number {i}"])
    print(f"wrote {len(CLIPS)} clips to {session}")


if __name__ == "__main__":
    main()
