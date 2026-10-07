"""Write data/processed/DATASET_CARD.md from what finalize.py produced.

Stats: hours and clip counts per language group, duration and SNR spread,
drops and flags by reason, train/val sizes. Provenance is fixed text below and
must stay exactly as recorded in my notes: no claims of edits that didn't happen.

Usage:  python data_prep/dataset_card.py [--processed data/processed]
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

PROVENANCE = """\
- **Speaker:** one adult male speaker (the author), recorded himself on a MacBook Air built-in mic, 48 kHz mono, in sessions of about 20 minutes. Consent: own voice.
- **Prompts p0001-p0480:** generated from templates by a script (`build_prompts.py`, written with AI assistance), then checked by script for spoken-form rules (no digits or symbols, acronyms as spaced letters).
- **Prompts p0481-p0650:** drafted by an AI assistant (8 Oct 2026) as a Hindi/Hinglish top-up, then checked by script for the same rules and for overlap with the benchmark.
- **Edits:** 24 prompts were changed by the AI assistant on the speaker's instruction to use masculine first-person forms, and two self-introductions with a name were made neutral (ids listed in the project notes). No other edits recorded as of 8 Oct 2026.
- **Benchmark and reference sentences** are kept out of training; finalize.py checks this (leakage.csv).
- **Access:** private. The audio is never published; the Kaggle dataset is private."""


def read(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def spread(values: list[float]) -> str:
    if not values:
        return "-"
    v = np.array(values)
    return f"median {np.median(v):.1f}, min {v.min():.1f}, max {v.max():.1f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed", default=str(ROOT / "data" / "processed"))
    out = Path(ap.parse_args().processed)
    clips = read(out / "keep_drop.csv")
    used = [c for c in clips if c["status"] != "drop"]

    lines = ["# Dataset card: desi-tts own-voice data", "", "## Provenance", "", PROVENANCE, "",
             "## Size (clips used for training, after trimming)", "",
             "| group | XTTS language | clips | hours | train | val |", "|---|---|---|---|---|---|"]
    for group in ("en", "hinglish", "hi"):
        g = [c for c in used if c["group"] == group]
        hours = sum(float(c["seconds"]) for c in g) / 3600
        lines.append(f"| {group} | {'hi' if group == 'hi' else 'en'} | {len(g)} | {hours:.2f} | "
                     f"{sum(c['split'] == 'train' for c in g)} | {sum(c['split'] == 'val' for c in g)} |")
    total = sum(float(c["seconds"]) for c in used) / 3600
    lines += [f"| **total** | | {len(used)} | {total:.2f} | | |", "",
              "## Quality", "",
              f"- Duration (s): {spread([float(c['seconds']) for c in used])}",
              f"- SNR (dB): {spread([float(c['snr_db']) for c in used])}",
              f"- Status: {dict(Counter(c['status'] for c in clips))}", "",
              "## Drops and flags by reason", "", "| reason | dropped | flagged |", "|---|---|---|"]
    by_reason: dict[str, Counter] = {}
    for c in clips:
        for reason in filter(None, c["reasons"].split("; ")):
            key = reason.split(" ")[0]
            by_reason.setdefault(key, Counter())[c["status"]] += 1
    for reason, n in sorted(by_reason.items()):
        lines.append(f"| {reason} | {n['drop']} | {n['flag']} |")
    lines += ["", "Recorded masters (48 kHz) are kept unchanged; training copies are trimmed, "
              "loudness-normalised to -23 LUFS and resampled to 22.05 kHz mono.", ""]
    (out / "DATASET_CARD.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
