"""Synthesise the benchmark with XTTS-v2 (pretrained or fine-tuned) and time it.

For every benchmark row it writes <id>.wav and one line in manifest.jsonl with
the text sent, language, audio length, wall time and RTF. With --stream it also
measures time to first audio (TTFA) using XTTS streaming.

The voice comes from --ref (one or more of my reference recordings; the
conditioning latents are computed once and reused for every sentence) or from
--speaker (a stock XTTS voice, only for testing the pipeline).

Every compared model must use the same --ref, --seed and configs/sampling.json.

Benchmark lang "hinglish" is sent to XTTS as "en" (Roman script), "hi" as "hi".

Examples (on Kaggle, through scripts/kaggle_run.sh):
  python infer/synthesize.py --model pretrained --speaker "Ana Florence" --sentence en "Hello there."
  python infer/synthesize.py --model pretrained --ref data/raw/reference/ref_en.wav --normalize off
  python infer/synthesize.py --model checkpoints/run1 --ref ref_en.wav --stream
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRETRAINED = "tts_models/multilingual/multi-dataset/xtts_v2"
XTTS_LANG = {"en": "en", "hinglish": "en", "hi": "hi"}


def read_benchmark(path: Path, limit: int | None) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return rows[:limit] if limit else rows


def model_dir(name: str) -> Path:
    """'pretrained' downloads XTTS-v2 (cached); anything else is a checkpoint folder."""
    if name != "pretrained":
        return Path(name)
    from TTS.utils.manage import ModelManager

    path = Path(ModelManager().download_model(PRETRAINED)[0])
    return path if path.is_dir() else path.parent


def load_model(folder: Path):
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    config = XttsConfig()
    config.load_json(str(folder / "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=str(folder), eval=True)
    return model.cuda()


def conditioning(model, refs: list[str] | None, speaker: str | None, s: dict):
    if speaker:
        return model.speaker_manager.speakers[speaker].values()
    return model.get_conditioning_latents(
        audio_path=refs,
        gpt_cond_len=s["gpt_cond_len"],
        gpt_cond_chunk_len=s["gpt_cond_chunk_len"],
        max_ref_length=s["max_ref_len"],
        sound_norm_refs=s["sound_norm_refs"],
    )


def prepare_text(text: str, normalize: bool) -> str:
    if not normalize:
        return text
    try:
        from text.normalize import normalize as norm  # built later (the Part 7 fix)
    except ImportError:
        sys.exit("--normalize on, but text/normalize.py doesn't exist yet. Use --normalize off.")
    return norm(text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="pretrained", help="'pretrained' or a fine-tuned checkpoint dir")
    ap.add_argument("--benchmark", default=str(ROOT / "benchmark" / "benchmark.tsv"))
    voice = ap.add_mutually_exclusive_group(required=True)
    voice.add_argument("--ref", nargs="+", help="reference wav(s) of the target voice")
    voice.add_argument("--speaker", help="stock XTTS speaker name (pipeline testing only)")
    ap.add_argument("--normalize", choices=["on", "off"], default="off")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--sampling", default=str(ROOT / "configs" / "sampling.json"))
    ap.add_argument("--stream", action="store_true", help="also measure time to first audio")
    ap.add_argument("--limit", type=int, help="only the first N rows (smoke tests)")
    ap.add_argument("--sentence", nargs=2, action="append", metavar=("LANG", "TEXT"),
                    help="use these sentences instead of the benchmark (smoke tests, repeatable)")
    ap.add_argument("--out", default=os.path.join(os.environ.get("OUT_DIR", "outputs"), "synth"))
    args = ap.parse_args()

    import soundfile as sf
    import torch

    settings = {k: v for k, v in json.load(open(args.sampling)).items() if not k.startswith("_")}
    if args.sentence:
        rows = [{"id": f"s{i:02d}", "lang": lang, "text": text} for i, (lang, text) in enumerate(args.sentence, 1)]
    else:
        rows = read_benchmark(Path(args.benchmark), args.limit)
    if not rows:
        sys.exit(f"{args.benchmark} has no sentences yet.")

    model = load_model(model_dir(args.model))
    sr = model.config.audio.output_sample_rate
    latent, embedding = conditioning(model, args.ref, args.speaker, settings)
    gen = {k: settings[k] for k in ("temperature", "length_penalty", "repetition_penalty",
                                    "top_k", "top_p", "speed", "enable_text_splitting")}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump({**vars(args), "sampling_used": settings}, open(out / "run_args.json", "w"), indent=2)

    # Warm-up so the first timed sentence doesn't pay one-off CUDA setup.
    model.inference("Warm up.", "en", latent, embedding, **gen)
    torch.cuda.reset_peak_memory_stats()

    with open(out / "manifest.jsonl", "w", encoding="utf-8") as manifest:
        for row in rows:
            lang = XTTS_LANG[row["lang"]]
            text = prepare_text(row["text"], args.normalize == "on")
            torch.manual_seed(args.seed)  # same seed per sentence, for every model

            torch.cuda.synchronize()
            start = time.perf_counter()
            wav = model.inference(text, lang, latent, embedding, **gen)["wav"]
            torch.cuda.synchronize()
            wall = time.perf_counter() - start

            seconds = len(wav) / sr
            sf.write(out / f"{row['id']}.wav", wav, sr)
            entry = {"id": row["id"], "lang": lang, "text_sent": text,
                     "seconds": round(seconds, 2), "wall": round(wall, 3),
                     "rtf": round(wall / seconds, 3) if seconds else None}

            if args.stream:
                torch.manual_seed(args.seed)
                torch.cuda.synchronize()
                start = time.perf_counter()
                stream = model.inference_stream(text, lang, latent, embedding, **gen)
                next(stream)
                torch.cuda.synchronize()
                entry["ttfa"] = round(time.perf_counter() - start, 3)
                for _ in stream:  # drain so the next sentence starts clean
                    pass

            manifest.write(json.dumps(entry, ensure_ascii=False) + "\n")
            print(entry, flush=True)

    print(f"peak VRAM {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    print(f"wrote {len(rows)} files to {out}")


if __name__ == "__main__":
    main()
