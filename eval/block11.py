"""Block 11 on the E3 weights: sampling sweep (E5) and long-sentence splitting.

E5  temperature 0.75 / 0.5 / 0.3  x  top_p 0.85 / 0.7  x  seeds 1-3, scored on
    the val clips (English + Hindi, Whisper CER) and the 10 held-out Hinglish
    sentences (Devanagari route, Devanagari CER), plus duration flags (babble).
    The benchmark is not used.
L   5 new long sentences (benchmark/long_heldout.tsv), split at sentence and
    clause boundaries (--split-long 120) vs not split, seeds 1-3.
All other settings as the final pipeline: my reference clips, normaliser on,
Hinglish via Devanagari. Jobs run two at a time, one per T4.

Usage (on Kaggle, KERNELS="desi-tts-e3-keep-extras", DATASETS incl. desi-tts-own-voice):
  python eval/block11.py --ref REF.wav ...
Writes $OUT_DIR/block11/{sweep,long}/... and sweep.csv / long.csv summaries.
"""
import argparse
import csv
import json
import os
import queue
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IN = Path("/tmp/in")
sys.path.insert(0, str(ROOT / "train"))
from select_checkpoint import val_set  # noqa: E402

TEMPS = ["0.75", "0.5", "0.3"]
TOP_PS = ["0.85", "0.7"]
SEEDS = ["1", "2", "3"]


def e3_model() -> str:
    sel = json.load(open(IN / "desi-tts-e3-keep-extras/out/e3/select/selected.json"))
    return str(IN / "desi-tts-e3-keep-extras/out/e3/select" / Path(sel["export"]).name)


def run_jobs(jobs: list[tuple[Path, list[str]]]) -> None:
    import torch

    free = queue.Queue()
    for gpu in range(max(1, torch.cuda.device_count())):
        free.put(gpu)

    def one(out: Path, cmd: list[str]) -> None:
        gpu = free.get()
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            with open(f"{out}.log", "w") as log:
                for script, extra in (("infer/synthesize.py", cmd), ("eval/asr_eval.py", None),
                                      ("eval/duration_check.py", None)):
                    if extra is None:
                        extra = ["--synth-dir", str(out)] + (["--benchmark", cmd[cmd.index("--benchmark") + 1],
                                                               "--deva", cmd[cmd.index("--benchmark") + 1]]
                                                              if script.endswith("asr_eval.py") else [])
                    subprocess.run([sys.executable, script, *extra], check=True, cwd=ROOT, env=env,
                                   stdout=log, stderr=subprocess.STDOUT)
            print(f"[gpu {gpu}] done {out}", flush=True)
        finally:
            free.put(gpu)

    with ThreadPoolExecutor(max_workers=free.qsize()) as pool:
        for f in [pool.submit(one, out, cmd) for out, cmd in jobs]:
            f.result()


def summary(folder: Path) -> dict:
    asr = list(csv.DictReader(open(folder / "asr_eval.csv", encoding="utf-8")))
    dur = list(csv.DictReader(open(folder / "duration_check.csv", encoding="utf-8")))
    cer = lambda rows: round(sum(float(r["cer"]) for r in rows) / len(rows), 4) if rows else None
    return {"cer_en_hi": cer([r for r in asr if r["lang"] != "hinglish"]),
            "cer_hinglish_deva": cer([r for r in asr if r["lang"] == "hinglish"]),
            "flags": sum(bool(r["flags"]) for r in dur), "n": len(asr)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", nargs="+", required=True)
    args = ap.parse_args()
    out = Path(os.environ.get("OUT_DIR", "outputs")) / "block11"
    out.mkdir(parents=True, exist_ok=True)
    model = e3_model()
    print("E3 model:", model, flush=True)

    val = val_set(IN / "desi-tts-own-voice", out)
    heldout = ROOT / "benchmark" / "hinglish_heldout.tsv"
    jobs, combos = [], []
    for t in TEMPS:
        for p in TOP_PS:
            cfg = {k: v for k, v in json.load(open(ROOT / "configs" / "sampling.json")).items()}
            cfg["temperature"], cfg["top_p"] = float(t), float(p)
            sampling = out / f"sampling_t{t}_p{p}.json"
            json.dump(cfg, open(sampling, "w"), indent=2)
            combos.append((t, p))
            for seed in SEEDS:
                for name, texts in (("val", val), ("heldout", heldout)):
                    jobs.append((out / "sweep" / f"t{t}_p{p}" / f"{name}_seed{seed}",
                                 ["--model", model, "--ref", *args.ref, "--benchmark", str(texts), "--normalize", "on",
                                  "--hinglish-route", "deva", "--seed", seed, "--sampling", str(sampling),
                                  "--out", str(out / "sweep" / f"t{t}_p{p}" / f"{name}_seed{seed}")]))
    long_texts = ROOT / "benchmark" / "long_heldout.tsv"
    for split in ("0", "120"):
        for seed in SEEDS:
            folder = out / "long" / f"split{split}_seed{seed}"
            jobs.append((folder, ["--model", model, "--ref", *args.ref, "--benchmark", str(long_texts),
                                  "--normalize", "on", "--hinglish-route", "deva", "--seed", seed,
                                  "--split-long", split, "--out", str(folder)]))
    print(f"{len(jobs)} jobs", flush=True)
    run_jobs(jobs)

    with open(out / "sweep.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["temperature", "top_p", "val_cer_en_hi", "heldout_cer_hinglish", "duration_flags_total"])
        for t, p in combos:
            v = [summary(out / "sweep" / f"t{t}_p{p}" / f"val_seed{s}") for s in SEEDS]
            h = [summary(out / "sweep" / f"t{t}_p{p}" / f"heldout_seed{s}") for s in SEEDS]
            w.writerow([t, p, round(sum(x["cer_en_hi"] for x in v) / 3, 4),
                        round(sum(x["cer_hinglish_deva"] for x in h) / 3, 4), sum(x["flags"] for x in v + h)])
    with open(out / "long.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split_chars", "cer_en_hi", "cer_hinglish_deva", "duration_flags_total"])
        for split in ("0", "120"):
            r = [summary(out / "long" / f"split{split}_seed{s}") for s in SEEDS]
            w.writerow([split, round(sum(x["cer_en_hi"] for x in r) / 3, 4),
                        round(sum(x["cer_hinglish_deva"] or 0 for x in r) / 3, 4), sum(x["flags"] for x in r)])
    print(open(out / "sweep.csv").read())
    print(open(out / "long.csv").read())


if __name__ == "__main__":
    main()
