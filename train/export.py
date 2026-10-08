"""Turn a training checkpoint into a folder infer/synthesize.py can load.

Training checkpoints hold the whole GPTTrainer state plus the optimizer
(several GB). This keeps only the model weights and adds the base XTTS files
inference needs (config.json, vocab.json, speakers_xtts.pth).
XTTS's loader drops the DVAE/mel keys and strips the "xtts." prefix itself.

Usage:  python train/export.py --run <training run dir> --out <export dir> [--checkpoint checkpoint_80.pth]
"""
import argparse
import re
import shutil
from pathlib import Path


def latest_checkpoint(run: Path) -> Path:
    found = sorted(run.rglob("checkpoint_*.pth"), key=lambda p: int(re.findall(r"\d+", p.stem)[-1]))
    if not found:
        raise SystemExit(f"no checkpoint_*.pth under {run}")
    return found[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", help="a file path, or a file name inside the run (default: highest step)")
    ap.add_argument("--base", default="/tmp/xtts_base", help="base XTTS files (from train_gpt.py)")
    args = ap.parse_args()

    import torch

    run, out, base = Path(args.run), Path(args.out), Path(args.base)
    if args.checkpoint and Path(args.checkpoint).exists():
        ckpt = Path(args.checkpoint)  # e.g. a milestones/step_N.pth file
    elif args.checkpoint:
        ckpt = next(run.rglob(args.checkpoint))
    else:
        ckpt = latest_checkpoint(run)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": state["model"], "step": state.get("step")}, out / "model.pth")
    for name in ("config.json", "vocab.json", "speakers_xtts.pth"):
        shutil.copy(base / name, out / name)
    size_mb = (out / "model.pth").stat().st_size / 1024**2
    print(f"exported {ckpt.name} (step {state.get('step')}) -> {out} ({size_mb:.0f} MB model.pth)")


if __name__ == "__main__":
    main()
