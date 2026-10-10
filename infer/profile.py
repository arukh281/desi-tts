"""Where the time goes: end-to-end latency of one model, stage by stage.

Stages per sentence: text normaliser, Hinglish transliteration (IndicXlit,
Hinglish rows only), GPT + vocoder (full clip), and streaming time to first
audio. Conditioning latents from the reference clips are computed once per
voice and timed separately (a server would cache them). Also reports model
size on disk, peak VRAM, and RTF.

Usage (on Kaggle):
  python infer/profile.py --model <export dir> --ref REF.wav ... [--benchmark FILE] [--repeat 2]
Writes $OUT_DIR/profile/profile.csv and profile.json.
"""
import argparse
import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "infer"))


def ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ref", nargs="+", required=True)
    ap.add_argument("--benchmark", default=str(ROOT / "benchmark" / "benchmark.tsv"))
    ap.add_argument("--chunk-size", type=int, default=20)
    ap.add_argument("--repeat", type=int, default=1, help="passes over the text set (first one includes warm-up effects)")
    args = ap.parse_args()

    import torch

    from synthesize import XTTS_LANG, conditioning, load_model, model_dir
    from text.normalize import normalize
    from text.translit import _translator, to_devanagari

    settings = {k: v for k, v in json.load(open(ROOT / "configs" / "sampling.json")).items() if not k.startswith("_")}
    gen = {k: settings[k] for k in ("temperature", "length_penalty", "repetition_penalty", "top_k", "top_p",
                                    "speed", "enable_text_splitting")}
    folder = model_dir(args.model)
    size_mb = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file()) / 1024**2

    t = time.perf_counter()
    model = load_model(folder)
    load_ms = ms(t)
    t = time.perf_counter()
    _translator()
    xlit_load_ms = ms(t)
    torch.cuda.synchronize()
    t = time.perf_counter()
    latent, embedding = conditioning(model, args.ref, None, settings)
    torch.cuda.synchronize()
    cond_ms = ms(t)
    sr = model.config.audio.output_sample_rate
    model.inference("Warm up.", "en", latent, embedding, **gen)
    torch.cuda.reset_peak_memory_stats()

    rows = list(csv.DictReader(open(args.benchmark, encoding="utf-8"), delimiter="\t"))
    out_rows = []
    for _ in range(args.repeat):
        for row in rows:
            t = time.perf_counter()
            text = normalize(row["text"], row["lang"])[0]
            norm_ms = ms(t)
            lang, xlit_ms = XTTS_LANG[row["lang"]], 0.0
            if row["lang"] == "hinglish":
                t = time.perf_counter()
                text, lang = to_devanagari(row["text"])[0], "hi"
                xlit_ms = ms(t)
            torch.manual_seed(1234)
            torch.cuda.synchronize()
            t = time.perf_counter()
            stream = model.inference_stream(text, lang, latent, embedding, stream_chunk_size=args.chunk_size, **gen)
            first = next(stream)
            torch.cuda.synchronize()
            ttfa_ms = ms(t)
            chunks = [first, *stream]
            torch.cuda.synchronize()
            total_ms = ms(t)
            audio_s = sum(len(c) for c in chunks) / sr
            out_rows.append({"id": row["id"], "lang": row["lang"], "chars": len(text), "normalise_ms": norm_ms,
                             "translit_ms": xlit_ms, "ttfa_ms": ttfa_ms, "synth_ms": total_ms,
                             "audio_s": round(audio_s, 2), "rtf": round(total_ms / 1000 / audio_s, 3)})
            print(out_rows[-1], flush=True)

    out = Path(os.environ.get("OUT_DIR", "outputs")) / "profile"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "profile.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    med = lambda key, rs=out_rows: round(statistics.median(float(r[key]) for r in rs), 2)
    p95 = lambda key: round(sorted(float(r[key]) for r in out_rows)[int(0.95 * (len(out_rows) - 1))], 2)
    hing = [r for r in out_rows if r["lang"] == "hinglish"]
    summary = {
        "gpu": torch.cuda.get_device_name(0), "model_mb_on_disk": round(size_mb, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "model_load_ms": load_ms, "indicxlit_load_ms": xlit_load_ms, "conditioning_once_ms": cond_ms,
        "n_sentences": len(out_rows), "stream_chunk_size": args.chunk_size,
        "normalise_ms_median": med("normalise_ms"), "translit_ms_median_hinglish": med("translit_ms", hing) if hing else None,
        "ttfa_ms_median": med("ttfa_ms"), "ttfa_ms_p95": p95("ttfa_ms"),
        "synth_ms_median": med("synth_ms"), "rtf_median": med("rtf"), "rtf_p95": p95("rtf"),
    }
    json.dump(summary, open(out / "profile.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
