"""Speaker consistency: spread across sentences, and sameness across languages. CPU.

1. Spread: per-sentence ECAPA similarity to the same-language reference clip,
   from the 3-seed results (results/final/*/seed*/speaker_sim.csv): mean, std
   and min per language. Final = E3 for en/hi, E3_deva for Hinglish.
2. Across languages: ECAPA embeddings of every clip, averaged per language,
   then cosine between the language averages (en-hi, en-hinglish, hi-hinglish).
   High = the same voice in all three languages. Run on the seed-1234 audio
   (E1 single run, E3 run 1) and, as a reference point, on my real takes.

Usage:
  python eval/consistency.py --e1 E1_WAV_DIR --final E3_WAV_DIR --final-hinglish E3_DEVA_WAV_DIR --real data/processed
Writes results/consistency.md.
"""
import argparse
import csv
import sys
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
LANGS = ("en", "hi", "hinglish")


def spread(systems: dict[str, dict[str, str]]) -> list[str]:
    lines = ["| system | language | n | mean | std | min |", "|---|---|---|---|---|---|"]
    for name, by_lang in systems.items():
        for lang in LANGS:
            v = [float(r["cosine"]) for f in (ROOT / "results" / "final" / by_lang[lang]).glob("seed*/speaker_sim.csv")
                 for r in csv.DictReader(open(f, encoding="utf-8")) if r["lang"] == lang]
            lines.append(f"| {name} | {lang} | {len(v)} | {statistics.mean(v):.3f} | {statistics.stdev(v):.3f} | {min(v):.3f} |")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--e1", type=Path, required=True)
    ap.add_argument("--final", type=Path, required=True)
    ap.add_argument("--final-hinglish", type=Path, required=True)
    ap.add_argument("--real", type=Path, required=True, help="processed data folder (clips.csv + wavs/)")
    args = ap.parse_args()

    import torch
    from speechbrain.inference.speaker import EncoderClassifier

    from speaker_sim import ENCODER, embed

    lines = ["<!-- python eval/consistency.py -->", "### Spread over sentences (3 seeds, vs same-language reference clip)", ""]
    lines += spread({"E1": {"en": "E1", "hi": "E1", "hinglish": "E1"},
                     "Final": {"en": "E3", "hi": "E3", "hinglish": "E3_deva"}})

    encoder = EncoderClassifier.from_hparams(source=ENCODER, savedir="/tmp/ecapa", run_opts={"device": "cpu"})
    bench = {r["id"]: r["lang"] for r in csv.DictReader(open(ROOT / "benchmark" / "benchmark.tsv", encoding="utf-8"), delimiter="\t")}
    sets = {"E1": {i: args.e1 / f"{i}.wav" for i in bench},
            "Final": {i: (args.final_hinglish if lang == "hinglish" else args.final) / f"{i}.wav" for i, lang in bench.items()}}
    used = {r["id"] for r in csv.DictReader(open(args.real / "clips.csv", encoding="utf-8")) if r["status"] != "drop"}
    real_lang = {r["id"]: r["group"] for r in csv.DictReader(open(args.real / "asr.csv", encoding="utf-8")) if r["id"] in used}
    sets["My real takes"] = {i: args.real / "wavs" / f"{i}.wav" for i in real_lang}
    lang_of = {"E1": bench, "Final": bench, "My real takes": real_lang}

    lines += ["", "### Same voice across languages (cosine between per-language average embeddings)", "",
              "| system | en-hi | en-hinglish | hi-hinglish |", "|---|---|---|---|"]
    for name, files in sets.items():
        by = defaultdict(list)
        for i, path in files.items():
            if path.exists():
                by[lang_of[name][i]].append(embed(encoder, str(path)))
        mean = {l: torch.stack(v).mean(dim=0) for l, v in by.items()}
        cos = lambda a, b: float(torch.nn.functional.cosine_similarity(mean[a], mean[b], dim=0))
        lines.append(f"| {name} (n {', '.join(f'{l} {len(by[l])}' for l in LANGS)}) | {cos('en', 'hi'):.3f} | "
                     f"{cos('en', 'hinglish'):.3f} | {cos('hi', 'hinglish'):.3f} |")
    text = "\n".join(lines) + "\n"
    (ROOT / "results" / "consistency.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
