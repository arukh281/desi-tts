"""Build the sample page: docs/index.html plus small MP3s in docs/audio/.

Baseline = E1 (pretrained XTTS-v2 + normaliser, cloned from my 3 reference
clips). Final = E3 run 1 (fine-tuned, step 1500) + normaliser, with Hinglish
through the Devanagari route. These are the same audio files the listening
test used. Seed 1234, one take each, not picked for being good: the five
sentences from the brief, two Hindi rows, three Hinglish rows, and three
name/place rows where the final model does worse.

The WAVs are private (outputs of the Kaggle runs); only the MP3s listed in
CLIPS are published. Needs ffmpeg.

Usage:
  python scripts/build_samples.py --e1 E1_WAV_DIR --e3 E3_NORM_WAV_DIR --e3-deva E3_DEVA_WAV_DIR
"""
import argparse
import csv
import html
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

GOOD = ["b005", "b031", "b037", "b045", "b077", "b080"]
HINGLISH = ["b061", "b063", "b076"]
FAILURES = ["b009", "b011", "b019"]
NOTES = {
    "b009": "Final model babbles: it adds a sentence that isn't there (Whisper: \"...Venkatesh here speaking? My parents would sound quite saluted speaking.\").",
    "b011": "\"Anjali Reddy\" comes out close to \"and daddy ready\".",
    "b019": "\"Guwahati\" is mispronounced (Whisper: \"Buahtai\"). The baseline says it better.",
}


def mp3(src: Path, name: str) -> str:
    out = DOCS / "audio" / f"{name}.mp3"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ac", "1", "-ar", "22050",
                    "-b:a", "40k", str(out)], check=True)
    return f"audio/{out.name}"


def cer(folder: Path) -> dict[str, float]:
    return {r["id"]: float(r["cer"]) for r in csv.DictReader(open(folder / "asr_eval.csv", encoding="utf-8"))}


def cell(label: str, path: str, score: float) -> str:
    return (f'<div class="take"><span class="who">{label}</span>'
            f'<audio controls preload="none" src="{path}"></audio>'
            f'<span class="cer">CER {score:.2f}</span></div>')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--e1", type=Path, required=True)
    ap.add_argument("--e3", type=Path, required=True)
    ap.add_argument("--e3-deva", type=Path, required=True)
    args = ap.parse_args()
    (DOCS / "audio").mkdir(parents=True, exist_ok=True)
    bench = {r["id"]: r for r in csv.DictReader(open(ROOT / "benchmark" / "benchmark.tsv", encoding="utf-8"), delimiter="\t")}
    runs = ROOT / "results" / "runs"
    c_e1, c_e3, c_deva = cer(runs / "E1_single"), cer(runs / "E3_run1" / "bench_norm_on"), cer(runs / "E3_run1" / "bench_deva")

    def row(i: str, takes: list[str]) -> str:
        b = bench[i]
        note = f'<p class="note">{html.escape(NOTES[i])}</p>' if i in NOTES else ""
        return (f'<section class="row"><p class="meta">{i} · {b["lang"]} · {b["category"].replace("_", " ")}</p>'
                f'<p class="text">{html.escape(b["text"])}</p>{note}<div class="takes">{"".join(takes)}</div></section>')

    def pair(i: str) -> str:
        return row(i, [cell("Baseline", mp3(args.e1 / f"{i}.wav", f"{i}_baseline"), c_e1[i]),
                       cell("Final", mp3(args.e3 / f"{i}.wav", f"{i}_final"), c_e3[i])])

    def hinglish(i: str) -> str:
        return row(i, [cell("Baseline (Roman)", mp3(args.e1 / f"{i}.wav", f"{i}_baseline"), c_e1[i]),
                       cell("Fine-tuned, Roman", mp3(args.e3 / f"{i}.wav", f"{i}_finetuned_roman"), c_e3[i]),
                       cell("Final (Devanagari)", mp3(args.e3_deva / f"{i}.wav", f"{i}_final"), c_deva[i])])

    page = (ROOT / "scripts" / "samples_template.html").read_text(encoding="utf-8")
    page = page.replace("{{GOOD}}", "".join(pair(i) for i in GOOD))
    page = page.replace("{{HINGLISH}}", "".join(hinglish(i) for i in HINGLISH))
    page = page.replace("{{FAILURES}}", "".join(pair(i) for i in FAILURES))
    (DOCS / "index.html").write_text(page, encoding="utf-8")
    size = sum(f.stat().st_size for f in (DOCS / "audio").glob("*.mp3"))
    print(f"{len(list((DOCS / 'audio').glob('*.mp3')))} MP3s, {size / 1024**2:.2f} MB")


if __name__ == "__main__":
    main()
