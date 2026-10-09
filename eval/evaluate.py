"""Synthesise a text set with one model and run all three evals on it, in one go.

Runs infer/synthesize.py, then eval/asr_eval.py, eval/speaker_sim.py and
eval/duration_check.py on the same output folder. Meant for one Kaggle run per
model, through scripts/kaggle_run.sh.

Usage:
  python eval/evaluate.py --name baseline --model pretrained --ref REF.wav [...] \\
      [--benchmark benchmark/benchmark.tsv] [--normalize on] [--stream]
  python eval/evaluate.py --name smoke --model pretrained --speaker "Ana Florence" \\
      --benchmark infer/fixture.tsv --sim-ref f02     (smoke: score against one of its own clips)
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> None:
    print("$", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], check=True, cwd=ROOT)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="output subfolder name")
    ap.add_argument("--model", default="pretrained")
    ap.add_argument("--benchmark", default=str(ROOT / "benchmark" / "benchmark.tsv"))
    voice = ap.add_mutually_exclusive_group(required=True)
    voice.add_argument("--ref", nargs="+")
    voice.add_argument("--speaker")
    ap.add_argument("--sim-ref", help="with --speaker: id of an output clip to use as the similarity reference")
    ap.add_argument("--normalize", choices=["on", "off"], default="off")
    ap.add_argument("--seed", default="1234")
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--synth-from", help="reuse audio from an earlier synth folder instead of synthesising "
                                         "(re-score only; --model/--normalize are then ignored)")
    args = ap.parse_args()

    out = Path(os.environ.get("OUT_DIR", "outputs")) / args.name
    synth = ["infer/synthesize.py", "--model", args.model, "--benchmark", args.benchmark,
             "--normalize", args.normalize, "--seed", args.seed, "--out", str(out)]
    synth += ["--ref", *args.ref] if args.ref else ["--speaker", args.speaker]
    synth += ["--stream"] if args.stream else []
    if args.synth_from:
        shutil.copytree(args.synth_from, out, dirs_exist_ok=True)
        print(f"reusing audio from {args.synth_from}")
    else:
        run(*synth)
    run("eval/asr_eval.py", "--synth-dir", str(out), "--benchmark", args.benchmark)
    if args.ref:
        sim_ref = args.ref
    elif args.sim_ref:
        sim_ref = [str(out / f"{args.sim_ref}.wav")]
    else:
        sim_ref = None
    if sim_ref:
        run("eval/speaker_sim.py", "--synth-dir", str(out), "--ref", *sim_ref)
    else:
        print("skipping speaker_sim: no --ref or --sim-ref")
    run("eval/duration_check.py", "--synth-dir", str(out))


if __name__ == "__main__":
    main()
