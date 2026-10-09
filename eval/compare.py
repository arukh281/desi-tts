"""Compare evaluated runs side by side and pick clips to listen to. No ML, runs on the Mac.

Each run folder is an eval/evaluate.py output (manifest.csv, asr_eval.csv,
speaker_sim.csv, duration_check.csv). Prints markdown tables:
  CER / WER per category and per language, speaker similarity per language,
  duration flags, median RTF and time to first audio.
Then lists the clips where run B beats run A most and loses most (by CER).

Usage:
  python eval/compare.py --run E0=path/e0 --run E1=path/e1 --run E2=path/e2 --listen E0,E2
"""
import argparse
import csv
import statistics
from pathlib import Path


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean(values) -> str:
    values = [float(v) for v in values]
    return f"{statistics.mean(values):.3f}" if values else "-"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="NAME=folder")
    ap.add_argument("--listen", help="A,B: list clips where B is most better / worse than A")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()
    runs = {k: Path(v) for k, v in (r.split("=", 1) for r in args.run)}
    data = {k: {"asr": read(p / "asr_eval.csv"), "sim": read(p / "speaker_sim.csv"),
                "dur": read(p / "duration_check.csv"), "man": read(p / "manifest.csv")} for k, p in runs.items()}
    names = list(runs)

    def table(title: str, key: str, rows_of) -> None:
        groups = sorted({r[key] for d in data.values() for r in d["asr"]})
        print(f"\n### {title}\n\n| {key} | n | " + " | ".join(f"{n} CER | {n} WER" for n in names) + " |")
        print("|---|---|" + "---|---|" * len(names))
        for g in groups:
            cells = []
            for n in names:
                rows = [r for r in data[n]["asr"] if r[key] == g]
                cells.append(f"{mean(r['cer'] for r in rows)} | {mean(r['wer'] for r in rows)}")
            count = len([r for r in data[names[0]]["asr"] if r[key] == g])
            print(f"| {g} | {count} | " + " | ".join(cells) + " |")

    table("Intelligibility by category (Whisper; Hinglish rows = rough Devanagari signal)", "category", None)
    table("Intelligibility by language", "lang", None)

    print("\n### Speaker similarity (ECAPA cosine vs the same-language reference clip)\n")
    langs = sorted({r["lang"] for d in data.values() for r in d["sim"]})
    print("| run | " + " | ".join(langs) + " |\n|---|" + "---|" * len(langs))
    for n in names:
        print(f"| {n} | " + " | ".join(mean(r["cosine"] for r in data[n]["sim"] if r["lang"] == l) for l in langs) + " |")

    print("\n### Timing and duration flags\n")
    print("| run | rows failed (XTTS error) | median RTF | median TTFA s | clips flagged | stretched | rushed | long pause | trailing | hyp longer |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for n in names:
        failed = sum(bool(r.get("error")) for r in data[n]["man"])
        man = [r for r in data[n]["man"] if not r.get("error")]
        dur = data[n]["dur"]
        rtf = statistics.median(float(r["rtf"]) for r in man if r["rtf"]) if man else 0
        ttfa = [float(r["ttfa_s"]) for r in man if r.get("ttfa_s")]
        count = lambda word: sum(word in r["flags"] for r in dur)
        print(f"| {n} | {failed} | {rtf:.3f} | {statistics.median(ttfa):.3f} | {sum(bool(r['flags']) for r in dur)} | "
              f"{count('stretched')} | {count('rushed')} | {count('long_pause')} | {count('trailing')} | {count('hyp_longer')} |"
              if ttfa else f"| {n} | {failed} | {rtf:.3f} | - | {sum(bool(r['flags']) for r in dur)} | | | | | |")

    if args.listen:
        a, b = args.listen.split(",")
        ca = {r["id"]: r for r in data[a]["asr"]}
        diffs = sorted((float(r["cer"]) - float(ca[r["id"]]["cer"]), r["id"]) for r in data[b]["asr"] if r["id"] in ca)
        print(f"\n### Clips to listen to: {b} vs {a}\n")
        for label, picks in ((f"{b} better", diffs[: args.n]), (f"{b} worse", diffs[::-1][: args.n])):
            for d, i in picks:
                print(f"- {label}: {i} (CER {ca[i]['cer']} -> {float(ca[i]['cer']) + d:.3f}) "
                      f"{runs[a] / (i + '.wav')}  vs  {runs[b] / (i + '.wav')}")


if __name__ == "__main__":
    main()
