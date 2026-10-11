"""Prosody on existing audio, CPU only: rate, comma pauses, sentence endings.

Systems: E1 and Final (seed-1234 audio: E1 single run, E3 run 1; Final uses the
Devanagari-route audio for Hinglish rows), and my real takes as a reference
point. The real takes are different sentences (training prompts), so compare
them loosely.

  rate      words per second of trimmed audio (spoken-form text, words counted
            on the normalised Roman/English/Devanagari text)
  pauses    silences of 120 ms or more inside speech (energy, 35 dB below peak),
            in sentences with commas: the long benchmark rows for E1/Final,
            takes with a comma and 12+ words for the real voice
  endings   pitch slope over the last 300 ms (eval/question_pitch.py): share of
            statements ("." endings) that fall, and of yes/no questions that rise

Usage:
  python eval/prosody.py --e1 E1_WAV_DIR --final E3_WAV_DIR --final-hinglish E3_DEVA_WAV_DIR --real data/processed
Writes results/prosody.md.
"""
import argparse
import csv
import re
import statistics
import sys
from pathlib import Path

import librosa

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
from question_pitch import YES_NO, end_slope  # noqa: E402

YES_NO_START = re.compile(r"^(is|are|can|could|would|will|do|does|did|should|have|has|kya|क्या)\b", re.I)


def trimmed(path: Path):
    y, sr = librosa.load(path, sr=16000)
    y, _ = librosa.effects.trim(y, top_db=35)
    return y, sr


def pauses_ms(path: Path) -> list[float]:
    y, sr = trimmed(path)
    spans = librosa.effects.split(y, top_db=35, frame_length=1024, hop_length=160)
    gaps = [(spans[k + 1][0] - spans[k][1]) / sr * 1000 for k in range(len(spans) - 1)]
    return [g for g in gaps if g >= 120]


def measure(items: list[dict]) -> dict:
    """items: {path, text, kind} with kind in statement/yesno/other."""
    rate = [len(it["text"].split()) / (len(trimmed(it["path"])[0]) / 16000) for it in items]
    comma = [it for it in items if it["comma"]]
    p = [g for it in comma for g in pauses_ms(it["path"])]
    commas = sum(it["text"].count(",") for it in comma)
    ends = {}
    for kind in ("statement", "yesno"):
        all_kind = [end_slope(it["path"]) for it in items if it["kind"] == kind]
        s = [v for v in all_kind if v is not None]
        ends[kind] = (sum(v < 0 for v in s) if kind == "statement" else sum(v > 0 for v in s), len(s))
        ends[kind + "_unvoiced"] = len(all_kind) - len(s)
    return {"rate": statistics.median(rate), "n": len(items), "pauses": len(p), "commas": commas,
            "comma_rows": len(comma), "pause_ms": statistics.median(p) if p else 0.0, "ends": ends}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--e1", type=Path, required=True)
    ap.add_argument("--final", type=Path, required=True)
    ap.add_argument("--final-hinglish", type=Path, required=True)
    ap.add_argument("--real", type=Path, required=True)
    args = ap.parse_args()

    bench = {r["id"]: r for r in csv.DictReader(open(ROOT / "benchmark" / "benchmark.tsv", encoding="utf-8"), delimiter="\t")}
    spoken = {r["id"]: r["text_used"] for r in csv.DictReader(open(args.e1 / "manifest.csv", encoding="utf-8"))}

    def kind(i: str, text: str) -> str:
        return "yesno" if i in YES_NO else "statement" if text.rstrip().endswith(".") else "other"

    def synth(folder_for) -> list[dict]:
        return [{"path": folder_for(b["lang"]) / f"{i}.wav", "text": spoken[i], "kind": kind(i, b["text"]),
                 "comma": b["category"] == "long"} for i, b in bench.items()]

    systems = {"E1": synth(lambda lang: args.e1),
               "Final": synth(lambda lang: args.final_hinglish if lang == "hinglish" else args.final)}
    used = {r["id"]: r for r in csv.DictReader(open(args.real / "clips.csv", encoding="utf-8")) if r["status"] != "drop"}
    systems["My real takes"] = [
        {"path": args.real / "wavs" / f"{i}.wav", "text": r["text"],
         "kind": "yesno" if r["text"].rstrip().endswith("?") and YES_NO_START.match(r["text"]) else
                 "statement" if r["text"].rstrip().endswith(".") else "other",
         "comma": "," in r["text"] and len(r["text"].split()) >= 12} for i, r in used.items() if (args.real / "wavs" / f"{i}.wav").exists()]

    res = {name: measure(items) for name, items in systems.items()}
    names = list(res)
    row = lambda label, f: f"| {label} | " + " | ".join(f(res[n]) for n in names) + " |"
    lines = ["<!-- python eval/prosody.py; seed-1234 audio (E1 single run, E3 run 1); real takes are different sentences -->",
             "| | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|",
             row("clips", lambda r: str(r["n"])),
             row("speaking rate, words/s (median)", lambda r: f"{r['rate']:.2f}"),
             row("comma sentences: commas / pauses ≥120 ms", lambda r: f"{r['commas']} / {r['pauses']} ({r['comma_rows']} sentences)"),
             row("median pause, ms", lambda r: f"{r['pause_ms']:.0f}"),
             row("statements ending with a fall", lambda r: "{}/{}".format(*r["ends"]["statement"])),
             row("yes/no questions ending with a rise", lambda r: "{}/{}".format(*r["ends"]["yesno"])),
             row("statements / questions left out (no voiced ending)",
                 lambda r: f"{r['ends']['statement_unvoiced']} / {r['ends']['yesno_unvoiced']}")]
    text = "\n".join(lines) + "\n"
    (ROOT / "results" / "prosody.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
