"""Time Hinglish -> Devanagari on this machine's CPU: cold, repeated, and new text.

Explains the Hinglish time-to-first-audio gap: transliteration runs before
synthesis, so its time adds to TTFA. Model load is done (and timed) first, so
no row includes it.
  cold:     the 16 benchmark Hinglish sentences, first time through
  repeat:   the same 16 again (every word now cached)
  new text: the 10 held-out sentences after that (only shared words cached)

Usage:  python infer/translit_timing.py   (writes results/translit_timing.json)
"""
import csv
import json
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from text.translit import _translator, to_devanagari  # noqa: E402


def rows(name: str) -> list[dict]:
    return [r for r in csv.DictReader(open(ROOT / "benchmark" / name, encoding="utf-8"), delimiter="\t")
            if r.get("lang", "hinglish") == "hinglish"]


def timed(rs: list[dict]) -> tuple[dict, list[str]]:
    ms, out = [], []
    for r in rs:
        t = time.perf_counter()
        out.append(to_devanagari(r["text"])[0])
        ms.append((time.perf_counter() - t) * 1000)
    return {"n": len(ms), "median_ms": round(statistics.median(ms), 2), "max_ms": round(max(ms), 2)}, out


def main() -> None:
    t = time.perf_counter()
    _translator()
    load_ms = round((time.perf_counter() - t) * 1000, 1)
    cold, first = timed(rows("benchmark.tsv"))
    repeat, second = timed(rows("benchmark.tsv"))
    new, _ = timed(rows("hinglish_heldout.tsv"))
    result = {"machine": f"{platform.system()} {platform.machine()}", "load_ms": load_ms, "cold": cold,
              "repeat": repeat, "new_text": new, "same_output_on_repeat": first == second}
    (ROOT / "results" / "translit_timing.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
