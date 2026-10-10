"""Final comparison run: E1, E2b and E3 on the full benchmark with 3 seeds, plus
the Hinglish 2x2 ({E1, E3} x {Roman/en, Devanagari/hi}) on the 16 benchmark rows
and on the 10 held-out sentences, also with 3 seeds. Same reference clips,
normaliser on, same eval settings throughout.

Fine-tuned weights come from earlier runs attached as kernel outputs
(KERNELS="desi-tts-e3-keep-extras desi-tts-e2b-keep").

Kaggle's "T4 x2" gives two GPUs. The jobs are independent, so they run two at
a time, one per GPU (nothing about any model or recipe changes).

Also a "real voice" row: my actual val recordings scored with the same
Whisper and speaker-similarity settings, i.e. what the metrics give real
speech. Needs DATASETS to include desi-tts-own-voice.

Usage (on Kaggle):  python eval/final_seeds.py --ref REF.wav ... [--seeds 1 2 3]
Output: $OUT_DIR/final/<system>/seed<k>/ with manifest + all eval CSVs.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IN = Path("/tmp/in")


def chosen_export(kernel: str, name: str) -> str:
    sel = json.load(open(IN / kernel / "out" / name / "select" / "selected.json"))
    return str(IN / kernel / "out" / name / "select" / Path(sel["export"]).name)


def evaluate(out: Path, gpu: int, *args: str) -> None:
    env = dict(os.environ, OUT_DIR=str(out.parent), CUDA_VISIBLE_DEVICES=str(gpu))
    log = out.parent / f"{out.name}.log"
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[gpu {gpu}] start {out.parent.name}/{out.name}", flush=True)
    with open(log, "w") as f:
        subprocess.run([sys.executable, "eval/evaluate.py", "--name", out.name, *args],
                       check=True, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    print(f"[gpu {gpu}] done  {out.parent.name}/{out.name}", flush=True)


def real_voice(final: Path, ref: list[str]) -> None:
    """My real val clips laid out like a synth folder, then the same evals."""
    import csv

    sys.path.insert(0, str(ROOT / "recording"))
    from order import language_group

    data = IN / "desi-tts-own-voice"
    if not data.exists():
        print("real voice: dataset not attached, skipped", flush=True)
        return
    out = final / "real_voice" / "seed1"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for lang in ("en", "hi"):
        meta = data / f"metadata_{lang}_val.csv"
        for r in csv.DictReader(open(meta, encoding="utf-8"), delimiter="|"):
            clip = Path(r["audio_file"]).stem
            group = language_group({"lang": lang, "text": r["text"]})
            link = out / f"{clip}.wav"
            if not link.exists():
                link.symlink_to(data / r["audio_file"])
            rows.append({"id": clip, "lang": group, "xtts_lang": lang, "text_in": r["text"], "text_used": r["text"]})
    with open(out / "manifest.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    for script, extra in (("eval/asr_eval.py", ["--deva", ""]), ("eval/speaker_sim.py", ["--ref", *ref])):
        subprocess.run([sys.executable, script, "--synth-dir", str(out), *extra], check=True, cwd=ROOT, env=env)
    print(f"real voice: {len(rows)} val clips scored", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", default=["1", "2", "3"])
    args = ap.parse_args()

    real_voice(Path(os.environ.get("OUT_DIR", "outputs")) / "final", args.ref)
    models = {"E1": "pretrained", "E3": chosen_export("desi-tts-e3-keep-extras", "e3")}
    if (IN / "desi-tts-e2b-keep").exists():  # E2b may run last (or not at all if GPU time runs out)
        models["E2b"] = chosen_export("desi-tts-e2b-keep", "e2b")
    else:
        print("E2b weights not attached: E2b stays at its single-seed run", flush=True)
    print("models:", models, flush=True)
    final = Path(os.environ.get("OUT_DIR", "outputs")) / "final"
    heldout = ROOT / "benchmark" / "hinglish_heldout.tsv"
    final.mkdir(parents=True, exist_ok=True)
    hing16 = final / "hinglish16.tsv"  # the 16 Hinglish benchmark rows; the Devanagari route changes only these
    lines = (ROOT / "benchmark" / "benchmark.tsv").read_text(encoding="utf-8").splitlines()
    hing16.write_text("\n".join([lines[0]] + [l for l in lines[1:] if l.split("\t")[1] == "hinglish"]) + "\n",
                      encoding="utf-8")
    jobs = []  # (output folder, extra args)
    for seed in args.seeds:
        for name, model in models.items():
            common = ["--model", model, "--ref", *args.ref, "--normalize", "on", "--seed", seed]
            jobs.append((final / name / f"seed{seed}", [*common, *(["--stream"] if seed == args.seeds[0] else [])]))
            if name in ("E1", "E3"):
                jobs.append((final / f"{name}_deva" / f"seed{seed}",
                             [*common, "--benchmark", str(hing16), "--hinglish-route", "deva"]))
                for route in ("en", "deva"):
                    jobs.append((final / f"{name}_heldout_{route}" / f"seed{seed}",
                                 [*common, "--benchmark", str(heldout), "--hinglish-route", route]))

    import queue
    from concurrent.futures import ThreadPoolExecutor

    import torch

    free = queue.Queue()
    for gpu in range(max(1, torch.cuda.device_count())):
        free.put(gpu)
    print(f"{len(jobs)} jobs on {free.qsize()} GPU(s)", flush=True)

    def run_on_free_gpu(out: Path, extra: list[str]) -> None:
        gpu = free.get()  # one job per GPU at a time
        try:
            evaluate(out, gpu, *extra)
        finally:
            free.put(gpu)

    with ThreadPoolExecutor(max_workers=free.qsize()) as pool:
        for f in [pool.submit(run_on_free_gpu, out, extra) for out, extra in jobs]:
            f.result()


if __name__ == "__main__":
    main()
