"""Final comparison run: E1, E2b and E3 on the full benchmark with 3 seeds, plus
the Hinglish 2x2 ({E1, E3} x {Roman/en, Devanagari/hi}) on the 16 benchmark rows
and on the 10 held-out sentences, also with 3 seeds. Same reference clips,
normaliser on, same eval settings throughout.

Fine-tuned weights come from earlier runs attached as kernel outputs
(KERNELS="desi-tts-e3-keep-extras desi-tts-e2b-keep").

Kaggle's "T4 x2" gives two GPUs. The jobs are independent, so they run two at
a time, one per GPU (nothing about any model or recipe changes).

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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", default=["1", "2", "3"])
    args = ap.parse_args()

    models = {"E1": "pretrained", "E2b": chosen_export("desi-tts-e2b-keep", "e2b"),
              "E3": chosen_export("desi-tts-e3-keep-extras", "e3")}
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

    import torch
    from concurrent.futures import ThreadPoolExecutor
    from itertools import cycle

    gpus = list(range(max(1, torch.cuda.device_count())))
    print(f"{len(jobs)} jobs on GPUs {gpus}", flush=True)
    slots = cycle(gpus)
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(evaluate, out, next(slots), *extra) for out, extra in jobs]
        for f in futures:
            f.result()


if __name__ == "__main__":
    main()
