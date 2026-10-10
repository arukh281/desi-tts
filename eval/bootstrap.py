"""Seed-averaged comparison of two systems with a paired bootstrap over sentences.

Each system folder holds seed*/ subfolders (eval/final_seeds.py output). Per
sentence, CER is averaged over seeds (a crashed row counts as CER 1.0). Then
the per-sentence difference B - A is resampled over sentences (10,000 times,
fixed seed) for a 95% interval, overall and per language. A difference counts
as real only if its interval excludes 0. Negative = B has lower CER (better).

Also prints, per category, each system's mean CER across seeds with the
min-max of the per-seed means as the spread.

Usage:  python eval/bootstrap.py --a E1=final/E1 --b E3=final/E3 [--metric cer]
"""
import argparse
import csv
import random
import statistics
from collections import defaultdict
from pathlib import Path


def load(folder: Path, metric: str) -> dict[str, dict[str, dict]]:
    """seed -> id -> {lang, category, value}; crashed rows score 1.0."""
    seeds = {}
    for seed_dir in sorted(folder.glob("seed*")):
        rows = {r["id"]: {"lang": r["lang"], "category": r["category"], "v": float(r[metric])}
                for r in csv.DictReader(open(seed_dir / "asr_eval.csv", encoding="utf-8"))}
        for r in csv.DictReader(open(seed_dir / "manifest.csv", encoding="utf-8")):
            if r.get("error") and r["id"] not in rows:
                rows[r["id"]] = {"lang": r["lang"], "category": "?", "v": 1.0}
        seeds[seed_dir.name] = rows
    return seeds


def per_sentence(seeds: dict) -> dict[str, float]:
    ids = set.intersection(*(set(s) for s in seeds.values()))
    return {i: statistics.mean(s[i]["v"] for s in seeds.values()) for i in ids}


def interval(diffs: list[float], n: int = 10_000) -> tuple[float, float, float]:
    rng = random.Random(0)
    boots = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(n))
    return statistics.mean(diffs), boots[int(0.025 * n)], boots[int(0.975 * n)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="NAME=folder")
    ap.add_argument("--b", required=True, help="NAME=folder")
    ap.add_argument("--metric", default="cer", choices=["cer", "wer"])
    args = ap.parse_args()
    (na, pa), (nb, pb) = (x.split("=", 1) for x in (args.a, args.b))
    sa, sb = load(Path(pa), args.metric), load(Path(pb), args.metric)
    ma, mb = per_sentence(sa), per_sentence(sb)
    meta = next(iter(sa.values()))
    common = sorted(set(ma) & set(mb))

    print(f"\n### {args.metric.upper()}: {nb} vs {na} (seeds: {len(sa)} vs {len(sb)}; {len(common)} sentences)\n")
    print("| subset | n | " + f"{na} | {nb} | {nb} - {na} | 95% interval | real? |")
    print("|---|---|---|---|---|---|---|")
    groups = {"all": common} | {f"lang={l}": [i for i in common if meta[i]["lang"] == l]
                                for l in sorted({meta[i]["lang"] for i in common})}
    for label, ids in groups.items():
        if not ids:
            continue
        diff, lo, hi = interval([mb[i] - ma[i] for i in ids])
        real = "yes" if lo > 0 or hi < 0 else "no"
        print(f"| {label} | {len(ids)} | {statistics.mean(ma[i] for i in ids):.3f} | "
              f"{statistics.mean(mb[i] for i in ids):.3f} | {diff:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {real} |")

    print(f"\n### {args.metric.upper()} per category: mean over seeds (min-max of per-seed means)\n")
    print(f"| category | n | {na} | {nb} |\n|---|---|---|---|")
    cats = defaultdict(list)
    for i in common:
        cats[meta[i]["category"]].append(i)
    for cat, ids in sorted(cats.items()):
        cells = []
        for seeds in (sa, sb):
            per_seed = [statistics.mean(s[i]["v"] for i in ids) for s in seeds.values()]
            cells.append(f"{statistics.mean(per_seed):.3f} ({min(per_seed):.3f}-{max(per_seed):.3f})")
        print(f"| {cat} | {len(ids)} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
