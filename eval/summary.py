"""One table for the report: CER and speaker similarity per system and language.

Reads results/final/<system>/seed*/ (asr_eval.csv, manifest.csv, speaker_sim.csv).
CER: per sentence averaged over seeds, crashed rows = 1.0 (same rule as
eval/bootstrap.py), then averaged over sentences. Speaker similarity: mean cosine
over all seeds and sentences. Writes results/summary.md.

Usage:  python eval/summary.py
"""
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
from bootstrap import load, per_sentence  # noqa: E402

LANGS = ("all", "en", "hi", "hinglish")


def speaker(folder: Path) -> dict[str, float]:
    by = defaultdict(list)
    for f in folder.glob("seed*/speaker_sim.csv"):
        for r in csv.DictReader(open(f, encoding="utf-8")):
            by[r["lang"]].append(float(r["cosine"]))
    return {k: round(statistics.mean(v), 3) for k, v in by.items()}


def main() -> None:
    lines = ["| system | seeds | n | " + " | ".join(f"CER {l}" for l in LANGS) + " | sim en | sim hi | sim hinglish |",
             "|---" * (3 + len(LANGS) + 3) + "|"]
    for folder in sorted((ROOT / "results" / "final").iterdir()):
        seeds = load(folder, "cer")
        cer = per_sentence(seeds)
        lang = next(iter(seeds.values()))
        cells = []
        for l in LANGS:
            v = [c for i, c in cer.items() if l == "all" or lang[i]["lang"] == l]
            cells.append(f"{statistics.mean(v):.3f} ({len(v)})" if v else "-")
        sim = speaker(folder)
        lines.append(f"| {folder.name} | {len(seeds)} | {len(cer)} | " + " | ".join(cells) + " | "
                     + " | ".join(f"{sim[l]:.3f}" if l in sim else "-" for l in ("en", "hi", "hinglish")) + " |")
    text = "\n".join(lines) + "\n"
    (ROOT / "results" / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
