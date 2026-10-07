"""Check that XTTS-v2 runs on a Kaggle T4 before we build anything on it.

Prints library versions, loads XTTS-v2 on the GPU, synthesises one English
and one Hindi sentence with a stock speaker, and reports wall time, RTF and
peak VRAM. Then it loads a WAV back with torchaudio, because voice cloning
reads reference recordings that way (torch 2.9+ needs torchcodec for it).

Run on Kaggle:  python scripts/env_check.py [--out DIR]
Default output dir: $OUT_DIR/env_check, or /kaggle/working/out/env_check.
"""
import argparse
import os
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

# XTTS-v2 is under the non-commercial CPML licence. Fine for this research
# take-home; setting this skips the interactive licence prompt.
os.environ["COQUI_TOS_AGREED"] = "1"

MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
SPEAKER = "Ana Florence"  # stock XTTS speaker; falls back to the first one listed

# Not in recording/prompts.tsv.
SENTENCES = [
    ("en", "Hello, I am checking that the speech model runs on this machine."),
    ("hi", "नमस्ते, यह जाँच है कि आवाज़ का मॉडल ठीक से चल रहा है।"),
]


def fail(msg: str) -> None:
    print(f"\nENV CHECK FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def pkg_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(os.environ.get("OUT_DIR", "/kaggle/working/out"), "env_check"))
    out_dir = Path(ap.parse_args().out)

    try:
        import soundfile as sf
        import torch
        from TTS.api import TTS
    except ImportError as e:
        fail(f"import error: {e}")

    print(f"python        {sys.version.split()[0]}")
    print(f"torch         {torch.__version__}")
    print(f"cuda          {torch.version.cuda}")
    print(f"coqui-tts     {pkg_version('coqui-tts')}")
    print(f"transformers  {pkg_version('transformers')}")

    if not torch.cuda.is_available():
        fail("no CUDA GPU found. In Kaggle, set Accelerator to GPU T4.")
    print(f"gpu           {torch.cuda.get_device_name(0)}\n")

    try:
        tts = TTS(MODEL).to("cuda")
    except Exception as e:
        fail(f"could not load {MODEL}: {e}")

    speakers = tts.speakers or []
    if not speakers:
        fail("model has no stock speakers")
    speaker = SPEAKER if SPEAKER in speakers else speakers[0]
    sr = tts.synthesizer.output_sample_rate
    print(f"speaker       {speaker}  ({sr} Hz)\n")

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Warm-up: the first call pays one-off CUDA setup costs, so don't time it.
        tts.tts(text="Warm up.", speaker=speaker, language="en")
        torch.cuda.reset_peak_memory_stats()

        for lang, text in SENTENCES:
            torch.cuda.synchronize()
            start = time.perf_counter()
            wav = tts.tts(text=text, speaker=speaker, language=lang)
            torch.cuda.synchronize()
            wall = time.perf_counter() - start

            duration = len(wav) / sr
            if duration == 0:
                fail(f"empty audio for {lang}")
            sf.write(out_dir / f"{lang}.wav", wav, sr)
            print(f"[{lang}] audio {duration:.2f}s  wall {wall:.2f}s  RTF {wall / duration:.2f}")
    except Exception as e:
        fail(f"synthesis error: {e}")

    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\npeak VRAM     {peak_gb:.2f} GB")

    try:
        import torchaudio

        audio, rate = torchaudio.load(str(out_dir / "en.wav"))
        print(f"torchaudio    {torchaudio.__version__}, loaded en.wav ({audio.shape[-1] / rate:.2f}s)")
    except Exception as e:
        fail(f"torchaudio could not load a wav (torchcodec missing or mismatched?): {e}")
    print(f"torchcodec    {pkg_version('torchcodec')}")
    print(f"saved to      {out_dir}")
    print("ENV CHECK PASSED")


if __name__ == "__main__":
    main()
