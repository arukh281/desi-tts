"""Fine-tune the GPT part of XTTS-v2 on my data. Everything else stays frozen.

Adapted from the coqui-tts 0.27.5 recipe recipes/ljspeech/xtts_v2/train_gpt_xtts.py
(same GPTArgs, audio config, optimizer and dataset handling). Changes from the
recipe: settings come from the command line, every non-GPT weight is frozen
explicitly and counted, there's one dataset per XTTS language (en, hi) with my
own val split, a run can resume from a previous run's folder, and train/val
loss is written to losses.csv.

Steps here are trainer steps (one batch each); with --grad-accum N the
optimizer updates once every N steps. The trainer only stops at epoch ends, so
--max-steps is turned into a number of epochs and the real count is printed.

Precision: fp32 by default. The T4 has fp16 tensor cores but no bf16, and fp16
without loss scaling care is known to give NaN losses when fine-tuning XTTS's
GPT. GPT-only fp32 with a batch of 2-4 fits in 16 GB. --fp16 is there to test.

Usage (on Kaggle, through scripts/kaggle_run.sh):
  python train/train_gpt.py --data DIR --out DIR [--lr 5e-6 --batch 2 --grad-accum 8 ...]
  python train/train_gpt.py --data DIR --out DIR --resume /kaggle/input/<previous run folder>
DIR holds wavs/ and metadata_{en,hi}_{train,val}.csv (data_prep/finalize.py output).
"""
import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

XTTS_FILES = "https://huggingface.co/coqui/XTTS-v2/resolve/main/"


def download_base(folder: Path) -> dict:
    """The recipe's base files: DVAE, mel stats, tokenizer, XTTS checkpoint (plus config/speakers for export)."""
    from TTS.utils.manage import ModelManager

    names = ["dvae.pth", "mel_stats.pth", "vocab.json", "model.pth", "config.json", "speakers_xtts.pth"]
    missing = [XTTS_FILES + n for n in names if not (folder / n).exists()]
    if missing:
        folder.mkdir(parents=True, exist_ok=True)
        ModelManager._download_model_files(missing, str(folder), progress_bar=False)
    return {n: str(folder / n) for n in names}


def datasets(data: Path) -> list:
    from TTS.config.shared_configs import BaseDatasetConfig

    out = []
    for lang in ("en", "hi"):
        train = data / f"metadata_{lang}_train.csv"
        if train.exists() and len(train.read_text(encoding="utf-8").splitlines()) > 1:
            out.append(BaseDatasetConfig(formatter="coqui", dataset_name=f"own_{lang}", path=str(data),
                                         meta_file_train=train.name, meta_file_val=f"metadata_{lang}_val.csv",
                                         language=lang))
    return out


def freeze_all_but_gpt(model) -> tuple[int, int]:
    trainable = total = 0
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("xtts.gpt.")
        total += p.numel()
        trainable += p.numel() if p.requires_grad else 0
    return trainable, total


class MilestoneSaver:
    """Keeps the model weights (no optimizer) at fixed fractions of training.

    About 2 GB each instead of ~5 GB for a full checkpoint, so all of them fit
    on Kaggle's disk and the val curve across training can be shown.
    """

    def __init__(self, folder: Path, steps: list[int]):
        self.folder, self.steps = folder, set(steps)
        folder.mkdir(parents=True, exist_ok=True)

    def on_train_step_end(self, trainer) -> None:
        import torch

        step = trainer.total_steps_done
        if step in self.steps:
            state = {k: v for k, v in trainer.model.state_dict().items() if k.startswith("xtts.")}
            torch.save({"model": state, "step": step}, self.folder / f"step_{step}.pth")
            print(f"milestone saved: step {step}")


class LossLog:
    """Writes step, losses and seconds/step to losses.csv as training runs."""

    def __init__(self, path: Path):
        self.path, self.last = path, None
        if not path.exists():
            path.write_text("step,split,loss,loss_text_ce,loss_mel_ce,sec_per_step,peak_vram_gb\n")

    def write(self, step: int, split: str, values: dict, sec: float | str = "") -> None:
        import torch

        def get(key: str) -> str:
            value = values.get(f"avg_{key}", values.get(key))
            return f"{float(value):.4f}" if value is not None else ""

        vram = f"{torch.cuda.max_memory_allocated() / 1024**3:.2f}"
        with open(self.path, "a") as f:
            f.write(f"{step},{split},{get('loss')},{get('loss_text_ce')},{get('loss_mel_ce')},{sec},{vram}\n")

    def on_train_step_end(self, trainer) -> None:
        now = time.perf_counter()
        sec = f"{now - self.last:.3f}" if self.last else ""
        self.last = now
        if trainer.keep_avg_train is not None:
            self.write(trainer.total_steps_done, "train", trainer.keep_avg_train.avg_values, sec)

    def on_epoch_end(self, trainer) -> None:
        if getattr(trainer, "keep_avg_eval", None) is not None:
            self.write(trainer.total_steps_done, "val", trainer.keep_avg_eval.avg_values)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder with wavs/ and metadata_<lang>_<split>.csv")
    ap.add_argument("--out", default=os.path.join(os.environ.get("OUT_DIR", "outputs"), "train"))
    ap.add_argument("--base", default="/tmp/xtts_base", help="where the base XTTS files are cached (not in outputs)")
    ap.add_argument("--resume", help="a previous run folder (read-only is fine; it gets copied)")
    ap.add_argument("--lr", type=float, default=5e-6, help="recipe default 5e-6")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--save-step", type=int, default=50)
    ap.add_argument("--keep", type=int, default=2, help="how many checkpoints to keep")
    ap.add_argument("--eval-step", type=int, default=50, help="run the val split every N steps")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--seed", type=int, default=1234, help="training seed (torch, numpy, random)")
    ap.add_argument("--milestones", default="0.25,0.5,0.75,1.0",
                    help="fractions of --max-steps at which to keep model weights (for checkpoint selection)")
    args = ap.parse_args()

    # Kaggle's "T4 x2" exposes two GPUs and the coqui trainer refuses to guess;
    # we train on one, so pin it before torch starts.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    import torch
    from trainer import Trainer, TrainerArgs
    from TTS.tts.datasets import load_tts_samples
    from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig
    from TTS.tts.models.xtts import XttsAudioConfig

    base = download_base(Path(args.base))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model_args = GPTArgs(
        max_conditioning_length=132300,  # 6 s
        min_conditioning_length=66150,  # 3 s
        debug_loading_failures=False,
        max_wav_length=255995,  # ~11.6 s at 22.05 kHz; data prep already drops longer clips
        max_text_length=200,
        mel_norm_file=base["mel_stats.pth"],
        dvae_checkpoint=base["dvae.pth"],
        xtts_checkpoint=base["model.pth"],
        tokenizer_file=base["vocab.json"],
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )
    config = GPTTrainerConfig(
        output_path=str(out),
        model_args=model_args,
        run_name="desi_tts_gpt",
        project_name="desi_tts",
        dashboard_logger="tensorboard",
        audio=XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000),
        batch_size=args.batch,
        batch_group_size=48,
        eval_batch_size=args.batch,
        num_loader_workers=2,  # Kaggle gives 4 CPUs
        print_step=10,
        plot_step=10,
        save_step=args.save_step,
        save_n_checkpoints=args.keep,
        save_checkpoints=True,
        save_best_after=10**9,  # no per-epoch best_model_*.pth: each is ~5 GB and fills Kaggle's disk
        run_eval=True,
        run_eval_steps=args.eval_step,
        print_eval=True,
        mixed_precision=args.fp16,
        precision="fp16",
        optimizer="AdamW",
        optimizer_wd_only_on_weights=True,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=args.lr,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [10**9], "gamma": 0.5, "last_epoch": -1},  # constant lr for short runs
        test_sentences=[],  # we evaluate with infer/ and eval/ instead
        training_seed=args.seed,
    )

    data = datasets(Path(args.data))
    if not data:
        raise SystemExit(f"no metadata_<lang>_train.csv with rows in {args.data}")
    train_samples, eval_samples = load_tts_samples(data, eval_split=True)
    steps_per_epoch = max(1, len(train_samples) // args.batch)
    target_epochs = math.ceil(args.max_steps / steps_per_epoch)
    config.epochs = target_epochs

    model = GPTTrainer.init_from_config(config)
    trainable, total = freeze_all_but_gpt(model)
    summary = {"trainable_params": trainable, "total_params": total, "train_clips": len(train_samples),
               "val_clips": len(eval_samples), "steps_per_epoch": steps_per_epoch, "epochs": config.epochs,
               "lr": args.lr, "batch": args.batch, "grad_accum": args.grad_accum, "fp16": args.fp16}
    print(json.dumps(summary, indent=2))

    continue_path = None
    if args.resume:
        run_dirs = [p for p in Path(args.resume).rglob("config.json") if any(p.parent.glob("checkpoint_*.pth"))]
        if not run_dirs:
            raise SystemExit(f"no checkpoint_*.pth with a config.json under {args.resume}")
        source = run_dirs[0].parent
        if os.access(source, os.W_OK):
            continue_path = source  # same session: keep training in place, no 5 GB copy
        else:
            continue_path = out / source.name  # e.g. /kaggle/input is read-only
            shutil.copytree(source, continue_path, dirs_exist_ok=True)
        print(f"resuming from {continue_path}")

    log = LossLog(out / "losses.csv")
    milestones = sorted({max(1, round(args.max_steps * float(f))) for f in args.milestones.split(",")})
    saver = MilestoneSaver(out / "milestones", milestones)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                         cwd=Path(__file__).resolve().parents[1]).stdout.strip()
    json.dump({**vars(args), "git_sha": sha, "milestone_steps": milestones,
               "train_clips": len(train_samples), "val_clips": len(eval_samples)},
              open(out / "run_config.json", "w"), indent=2)

    def on_step_end(trainer) -> None:
        log.on_train_step_end(trainer)
        saver.on_train_step_end(trainer)

    trainer = Trainer(
        TrainerArgs(restore_path=None, continue_path=str(continue_path) if continue_path else None,
                    skip_train_epoch=False, start_with_eval=False, grad_accum_steps=args.grad_accum),
        config, output_path=str(out), model=model,
        train_samples=train_samples, eval_samples=eval_samples,
        callbacks={"on_train_step_end": on_step_end, "on_epoch_end": log.on_epoch_end},
    )
    # On resume the trainer loads the old run's config.json into the same config
    # object, epoch count included, so set this run's target again from our own copy.
    trainer.config.epochs = target_epochs
    print(f"starting at step {trainer.total_steps_done}, epoch {trainer.epochs_done}, target epochs {target_epochs}")
    trainer.fit()
    summary.update({"steps_done": trainer.total_steps_done, "run_dir": str(trainer.output_path),
                    "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2)})
    json.dump(summary, open(out / "train_summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
