"""Block 15: inference-only fixes for names, on the final model (E3 + normaliser + Devanagari route).

Stage "heldout": tune on benchmark/names_heldout.tsv only (20 new English
sentences with Indian names and places), seeds 1-3, five settings:
  base      the final pipeline as it is
  temp      temperature 0.3 on name rows (everything else unchanged)
  respell   configs/respell_names.tsv applied to English text
  guard     babble guard: retry long clips with another seed, keep the shortest
  all       respell + temp + guard
Stage "bench": the full benchmark, seeds 1-3, with the settings named in --flags
(the pick from stage 1), run once.

Usage (on Kaggle, KERNELS="desi-tts-e3-keep-extras"):
  python eval/block15.py --stage heldout --ref REF.wav ...
  python eval/block15.py --stage bench --name <pick> --flags "<synthesize.py flags>" --ref REF.wav ...
Writes $OUT_DIR/block15/<stage>/<setting>/seed<k>/ (manifest + asr_eval + duration_check).
"""
import argparse
import os
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
from block11 import e3_model, run_jobs  # noqa: E402

SEEDS = ["1", "2", "3"]
RESPELL = str(ROOT / "configs" / "respell_names.tsv")
SETTINGS = {
    "base": [],
    "temp": ["--name-temperature", "0.3"],
    "respell": ["--respell", RESPELL],
    "guard": ["--babble-guard"],
    "all": ["--respell", RESPELL, "--name-temperature", "0.3", "--babble-guard"],
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["heldout", "bench"], required=True)
    ap.add_argument("--ref", nargs="+", required=True)
    ap.add_argument("--name", default="pick")
    ap.add_argument("--flags", default="")
    ap.add_argument("--model", help="model export dir (default: the final E3 weights)")
    ap.add_argument("--only", help="comma-separated settings to run in the heldout stage")
    args = ap.parse_args()
    out = Path(os.environ.get("OUT_DIR", "outputs")) / "block15" / args.stage
    model = args.model or e3_model()
    if args.stage == "heldout":
        texts, settings = ROOT / "benchmark" / "names_heldout.tsv", SETTINGS
        if args.only:
            settings = {k: v for k, v in SETTINGS.items() if k in args.only.split(",")}
    else:
        texts, settings = ROOT / "benchmark" / "benchmark.tsv", {args.name: shlex.split(args.flags.replace("RESPELL", RESPELL))}
    jobs = []
    for name, flags in settings.items():
        for seed in SEEDS:
            folder = out / name / f"seed{seed}"
            jobs.append((folder, ["--model", model, "--ref", *args.ref, "--benchmark", str(texts),
                                  "--normalize", "on", "--hinglish-route", "deva", "--seed", seed,
                                  *flags, "--out", str(folder)]))
    print(f"{len(jobs)} jobs", flush=True)
    run_jobs(jobs, deva=str(ROOT / "benchmark" / "hinglish_deva.tsv") if args.stage == "bench" else None)


if __name__ == "__main__":
    main()
