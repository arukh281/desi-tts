"""Step 3 of data prep: decide what trains, split it, and write XTTS metadata.

Reads data/processed/clips.csv (from prepare.py) and data/processed/asr.csv
(from asr_check.py, Whisper on Kaggle), then writes into data/processed/:
  keep_drop.csv     every clip with status keep / flag / drop and all reasons
  drift.csv         RMS and noise floor in take order, per session
  metadata_<lang>_train.csv, metadata_<lang>_val.csv
                    coqui "coqui" format (audio_file|text|speaker_name), one
                    pair per XTTS language: en (English + Hinglish) and hi
  leakage.csv       any train/val text too close to benchmark or reference text

Flagged clips stay in training; they're listed so I can listen to them.
The split is stratified by language group (en / hinglish / hi), fixed seed.

Usage:
  python data_prep/finalize.py
  python data_prep/finalize.py --no-asr     (before the Whisper check has run)
"""
import argparse
import csv
import json
import random
import sys
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "recording"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "data_prep"))
sys.path.insert(0, str(ROOT / "eval"))

from asr_common import cer as char_error_rate  # noqa: E402
from asr_common import score_text  # noqa: E402
from check_overlap import NEAR_DUP, normalise  # noqa: E402
from order import language_group  # noqa: E402
from prepare import CONFIG, load_config  # noqa: E402

SPEAKER = "speaker1"
XTTS_LANG = {"en": "en", "hinglish": "en", "hi": "hi"}


def read_csv(path: Path, delimiter: str = ",") -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def write_csv(path: Path, rows: list[dict], fields: list[str], delimiter: str = ",") -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter=delimiter, extrasaction="ignore",
                           lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def apply_asr(clip: dict, asr: dict | None, cfg: dict) -> None:
    """Add the transcript check to a clip's status and reasons."""
    reasons = [r for r in clip["reasons"].split("; ") if r]
    status = clip["status"]
    if asr is None:
        reasons.append("asr_not_run")
        status = "flag" if status == "keep" else status
    else:
        # Re-score from Whisper's raw text with the current normaliser, so a
        # scoring fix doesn't need another GPU run.
        group = clip["group"]
        cer = char_error_rate(score_text(clip["text"], group), score_text(asr["hypothesis"], group))
        clip["cer"], clip["asr_text"] = f"{cer:.3f}", asr["hypothesis"]
        if cfg["cer_drop"][group] is None:
            pass  # no reliable transcript check for this group (see config.json)
        elif cer > cfg["cer_drop"][group]:
            reasons.append(f"transcript_mismatch cer {cer:.2f}")
            status = "drop"
        elif cfg["cer_flag"][group] is not None and cer > cfg["cer_flag"][group]:
            reasons.append(f"transcript_borderline cer {cer:.2f}")
            status = "flag" if status == "keep" else status
    clip["status"], clip["reasons"] = status, "; ".join(reasons)


def drift(clips: list[dict]) -> list[dict]:
    rows = sorted(clips, key=lambda c: (c["session"], int(c["take_order"])))
    for session in sorted({c["session"] for c in rows}):
        takes = [c for c in rows if c["session"] == session]
        q = max(1, len(takes) // 4)
        first = [float(c["rms_dbfs"]) for c in takes[:q]]
        last = [float(c["rms_dbfs"]) for c in takes[-q:]]
        nf_first = [float(c["noise_floor_dbfs"]) for c in takes[:q]]
        nf_last = [float(c["noise_floor_dbfs"]) for c in takes[-q:]]
        print(f"drift {session}: RMS first quarter {sum(first) / len(first):.1f} -> last {sum(last) / len(last):.1f} dBFS, "
              f"noise floor {sum(nf_first) / len(nf_first):.1f} -> {sum(nf_last) / len(nf_last):.1f} dBFS")
    return rows


def split(clips: list[dict], fraction: float, seed: int) -> None:
    """Mark each usable clip train or val, stratified by language group."""
    rng = random.Random(seed)
    for group in ("en", "hinglish", "hi"):
        members = sorted((c for c in clips if c["group"] == group), key=lambda c: c["id"])
        rng.shuffle(members)
        n_val = round(len(members) * fraction) if len(members) >= 10 else 0
        for i, clip in enumerate(members):
            clip["split"] = "val" if i < n_val else "train"


def leakage(clips: list[dict], protected: list[str]) -> list[dict]:
    hits = []
    norm_protected = [(t, normalise(t)) for t in protected]
    for clip in clips:
        text = normalise(clip["text"])
        for original, other in norm_protected:
            ratio = SequenceMatcher(None, text, other).ratio()
            if ratio >= NEAR_DUP:
                hits.append({"id": clip["id"], "text": clip["text"], "matches": original, "ratio": round(ratio, 2)})
    return hits


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed", default=str(ROOT / "data" / "processed"))
    ap.add_argument("--benchmark", default=str(ROOT / "benchmark" / "benchmark.tsv"))
    ap.add_argument("--reference", default=str(ROOT / "recording" / "reference.txt"))
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--no-asr", action="store_true", help="run before the Whisper check; every clip gets flagged")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    out = Path(args.processed)
    clips = read_csv(out / "clips.csv")
    asr = {}
    if not args.no_asr:
        if not (out / "asr.csv").exists():
            sys.exit("asr.csv missing: run data_prep/asr_check.py on Kaggle first, or pass --no-asr")
        asr = {r["id"]: r for r in read_csv(out / "asr.csv")}

    for clip in clips:
        clip["group"] = language_group(clip)
        clip["xtts_lang"] = XTTS_LANG[clip["group"]]
        clip["cer"], clip["asr_text"], clip["split"] = "", "", ""
        if "|" in clip["text"]:
            clip["status"], clip["reasons"] = "drop", "pipe_in_text"
        apply_asr(clip, asr.get(clip["id"]), cfg)

    usable = [c for c in clips if c["status"] != "drop"]
    split(usable, cfg["val_fraction"], cfg["seed"])

    protected = [r["text"] for r in read_csv(Path(args.benchmark), "\t")]
    protected += [r["text"] for r in read_csv(Path(args.reference), "\t")]
    leaks = leakage(usable, protected)
    write_csv(out / "leakage.csv", leaks, ["id", "text", "matches", "ratio"])

    fields = ["id", "session", "take_order", "lang", "group", "xtts_lang", "status", "split", "reasons",
              "seconds", "snr_db", "cer", "text", "asr_text"]
    write_csv(out / "keep_drop.csv", clips, fields)
    write_csv(out / "drift.csv", drift(clips), ["session", "take_order", "id", "rms_dbfs", "noise_floor_dbfs"])

    for lang in ("en", "hi"):
        for part in ("train", "val"):
            rows = [{"audio_file": f"wavs/{c['id']}.wav", "text": c["text"], "speaker_name": SPEAKER}
                    for c in usable if c["xtts_lang"] == lang and c["split"] == part]
            write_csv(out / f"metadata_{lang}_{part}.csv", rows, ["audio_file", "text", "speaker_name"], "|")

    counts = {s: sum(c["status"] == s for c in clips) for s in ("keep", "flag", "drop")}
    print(f"{len(clips)} clips {counts}; train {sum(c['split'] == 'train' for c in usable)}, "
          f"val {sum(c['split'] == 'val' for c in usable)}; leakage hits {len(leaks)}")
    if leaks:
        sys.exit("LEAKAGE: training text overlaps benchmark/reference, see leakage.csv")


if __name__ == "__main__":
    main()
