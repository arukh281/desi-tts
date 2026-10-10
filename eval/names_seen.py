"""Name and place words that Whisper doesn't find in the transcript, seen vs unseen.

"Seen" means the name appears in my training prompts (benchmark tags column).
For each benchmark row in the names/places categories, every word of the
names in the check column counts as missed if it isn't in the Whisper
transcript (lowercased; Devanagari compared as is). Acronym and time entries
("KYC = K Y C") are skipped.

Usage:  python eval/names_seen.py NAME=RUN_DIR[,RUN_DIR...] ...   (several dirs, e.g. seeds, are pooled)
"""
import csv
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    bench = {r["id"]: r for r in csv.DictReader(open(ROOT / "benchmark" / "benchmark.tsv", encoding="utf-8"), delimiter="\t")}
    print("| run | type | seen missed | unseen missed |\n|---|---|---|---|")
    for arg in sys.argv[1:]:
        name, dirs = arg.split("=", 1)
        miss, total = Counter(), Counter()
        rows = [r for d in dirs.split(",") for r in csv.DictReader(open(Path(d) / "asr_eval.csv", encoding="utf-8"))]
        for r in rows:
            b = bench[r["id"]]
            if b["category"] not in ("names", "places"):
                continue
            seen = "seen" if "seen" in b["tags"].split(",") else "unseen"
            hyp = set(re.findall(r"\w+", r["hypothesis"].lower()))
            for entry in b["check"].split(";"):
                if "=" in entry:
                    continue
                for word in re.findall(r"\w+", entry.lower()):
                    total[b["category"], seen] += 1
                    miss[b["category"], seen] += word not in hyp
        for cat in ("names", "places"):
            cells = [f"{miss[cat, s]}/{total[cat, s]} = {miss[cat, s] / total[cat, s]:.2f}" if total[cat, s] else "-"
                     for s in ("seen", "unseen")]
            print(f"| {name} | {cat} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
