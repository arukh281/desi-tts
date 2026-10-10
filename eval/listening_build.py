"""Build the blind A/B listening test: one static page with the audio inside it.

22 pairs: 16 of the pretrained model (E1) vs the fine-tuned model (E3), both
with the normaliser, across categories; 4 Hinglish pairs of E3 reading Roman
text through "en" vs Devanagari through "hi"; and 2 earlier pairs repeated with
A and B swapped, to see if listeners answer consistently.

Which system is A and the order of the pairs come from a fixed seed. The page
never says which is which; the key goes to a separate file that stays private.

Usage:
  python eval/listening_build.py --audio ROOT --benchmark benchmark/benchmark.tsv \\
      --page OUT.html --key KEY.json
ROOT has e1/<id>.wav, e3_norm/<id>.wav and e3_deva/<id>.wav.
"""
import argparse
import base64
import csv
import html
import io
import json
import random
from pathlib import Path

import soundfile as sf

SEED = 20261010
E1_VS_E3 = ["b002", "b007", "b011", "b014", "b018", "b024", "b029", "b031",
            "b034", "b037", "b038", "b045", "b054", "b059", "b078", "b085"]
HINGLISH = ["b063", "b066", "b070", "b075"]
REPEATS = ["b037", "b078"]


def mp3_data_uri(path: Path) -> str:
    audio, sr = sf.read(path, dtype="float32")
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="MP3")
    return "data:audio/mpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--page", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--template", default=str(Path(__file__).with_name("listening_page.html")))
    args = ap.parse_args()

    root = Path(args.audio)
    text = {r["id"]: r for r in csv.DictReader(open(args.benchmark, encoding="utf-8"), delimiter="\t")}
    rng = random.Random(SEED)

    pairs = [{"row": i, "kind": "e1_vs_e3", "x": ("E1", root / "e1" / f"{i}.wav"),
              "y": ("E3", root / "e3_norm" / f"{i}.wav")} for i in E1_VS_E3]
    pairs += [{"row": i, "kind": "hinglish_roman_vs_deva", "x": ("E3_roman_en", root / "e3_norm" / f"{i}.wav"),
               "y": ("E3_deva_hi", root / "e3_deva" / f"{i}.wav")} for i in HINGLISH]
    for p in pairs:
        p["a_is_x"] = rng.random() < 0.5
    rng.shuffle(pairs)
    for row in REPEATS:  # same clips, A and B swapped, at least 6 pairs after the original
        at = next(i for i, p in enumerate(pairs) if p["row"] == row)
        spot = rng.randint(min(at + 7, len(pairs)), len(pairs))
        pairs.insert(spot, {**pairs[at], "a_is_x": not pairs[at]["a_is_x"], "repeat_of": row})

    key, cards = {}, []
    for n, p in enumerate(pairs, 1):
        pid = f"P{n:02d}"
        a, b = (p["x"], p["y"]) if p["a_is_x"] else (p["y"], p["x"])
        key[pid] = {"row": p["row"], "kind": p["kind"], "A": a[0], "B": b[0], "repeat_of": p.get("repeat_of")}
        cards.append({"id": pid, "text": text[p["row"]]["text"], "a": mp3_data_uri(a[1]), "b": mp3_data_uri(b[1])})

    page = Path(args.template).read_text(encoding="utf-8")
    page = page.replace("/*PAIRS*/[]", json.dumps(cards, ensure_ascii=False))
    Path(args.page).write_text(page, encoding="utf-8")
    json.dump({"seed": SEED, "pairs": key}, open(args.key, "w"), indent=2)
    size = Path(args.page).stat().st_size / 1024**2
    print(f"{len(cards)} pairs -> {args.page} ({size:.1f} MB); key -> {args.key}")
    for pid, k in key.items():
        print(pid, k["row"], k["kind"], "A =", k["A"], "(repeat)" if k["repeat_of"] else "")
    visible = page.split("const PAIRS =")[0] + json.dumps([{"id": c["id"], "text": c["text"]} for c in cards])
    leaks = [w for w in ("E1", "E3", "pretrained", "fine-tuned", "Devanagari", "Roman") if w in visible]
    print("system names outside the audio data:", leaks or "none")


if __name__ == "__main__":
    main()
