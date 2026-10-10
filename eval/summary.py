"""One table for the report: CER and speaker similarity per system and language.

Reads results/final/<system>/seed*/ (asr_eval.csv, manifest.csv, speaker_sim.csv).
CER: per sentence averaged over seeds, crashed rows = 1.0 (same rule as
eval/bootstrap.py), then averaged over sentences. Speaker similarity: mean cosine
over all seeds and sentences. Writes results/summary.md.

Also writes results/training.md: per training run, clips, steps, the val
selection table's pick, peak VRAM and training time (sum of seconds per step
from losses.csv, so setup and evaluation are not included).

Usage:  python eval/summary.py
"""
import csv
import json
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
    train = training()
    (ROOT / "results" / "training.md").write_text(train)
    print(train)


def training() -> str:
    lines = ["| run | train / val clips | steps | picked step | median s/step | training min | peak VRAM GB |",
             "|---|---|---|---|---|---|---|"]
    for run in sorted((ROOT / "results" / "runs").iterdir()):
        summary, losses = run / "train" / "train_summary.json", run / "train" / "losses.csv"
        if not (summary.exists() and losses.exists()):
            continue
        t = json.loads(summary.read_text())
        secs = [float(r["sec_per_step"]) for r in csv.DictReader(open(losses)) if r["split"] == "train" and r["sec_per_step"]]
        picked = json.loads((run / "select" / "selected.json").read_text()).get("step", "?")
        lines.append(f"| {run.name} | {t['train_clips']} / {t['val_clips']} | {t['steps_done']} | {picked} | "
                     f"{statistics.median(secs):.2f} | {sum(secs) / 60:.1f} | {t['peak_vram_gb']} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
