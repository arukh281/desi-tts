"""Record the training voice, one prompt at a time. Runs on the Mac, no ML.

Usage:
  python recording/record.py --session 1                 record prompts in recording/order.tsv
  python recording/record.py --session 1 --order recording/prompts.tsv
  python recording/record.py --reference                 record the 3 reference clips
  python recording/record.py --dry-run                   test everything with a fake signal, no mic

Keys while recording a prompt:
  Enter  start, Enter again to stop
  then:  Enter keep   r redo   s skip   q quit

It resumes on its own: any prompt that already has a WAV in data/raw/session_*/
is skipped, so you can quit and come back (even in a later session).

Each take is saved as a 48 kHz mono 24-bit WAV, data/raw/session_<n>/<id>.wav.
We downsample later; recording higher than needed costs nothing.
"""
import argparse
import csv
import json
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
ROMAN = ROOT / "recording" / "roman.tsv"  # private reading aid, see load_roman()
SR = 48_000
BUILT_IN_MIC = "MacBook"  # Bluetooth mics drop to phone-call quality, so we want the built-in one

CLIP_PEAK = 0.99  # at or above this the take is clipped
QUIET_PEAK = 0.05  # below this (about -26 dBFS) the take is too quiet
MAX_SECONDS = 11.6  # XTTS GPT recipe default max_wav_length=255995 samples at 22.05 kHz


# ---------- plain logic (unit-tested, no audio hardware) ----------

def read_tsv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def recorded_ids(raw_dir: Path) -> set[str]:
    """Prompt ids that already have a WAV in any session folder."""
    return {p.stem for p in raw_dir.glob("session_*/*.wav")}


def load_roman(path: Path) -> dict[str, str]:
    """Optional reading aid: Hindi lines in English letters (id -> roman).

    The saved transcript stays in Devanagari, which XTTS's Hindi mode expects;
    this only changes what's shown on screen.
    """
    if not path.exists():
        return {}
    return {r["id"]: r["roman"] for r in read_tsv(path)}


def todo(prompts: list[dict], done: set[str]) -> list[dict]:
    return [p for p in prompts if p["id"] not in done]


def peak_db(block: np.ndarray) -> float:
    peak = float(np.abs(block).max()) if block.size else 0.0
    return 20 * np.log10(peak + 1e-9)


def meter_bar(db: float, width: int = 30) -> str:
    """Text level meter from -60 dBFS (empty) to 0 dBFS (full)."""
    filled = int(round((min(max(db, -60.0), 0.0) + 60) / 60 * width))
    label = "CLIP" if db >= 20 * np.log10(CLIP_PEAK) else f"{db:5.1f} dB"
    return f"[{'#' * filled}{'.' * (width - filled)}] {label}"


def check_take(audio: np.ndarray, sr: int = SR) -> list[str]:
    """Warnings for a finished take, so bad ones get redone now."""
    warnings = []
    seconds = len(audio) / sr
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    if audio.size and peak == 0.0:
        warnings.append("pure silence: check System Settings > Privacy > Microphone for this terminal")
    elif peak >= CLIP_PEAK:
        warnings.append("clipped (too loud): move back a little and redo")
    elif peak < QUIET_PEAK:
        warnings.append("very low level: move closer or speak up")
    if seconds > MAX_SECONDS:
        warnings.append(f"{seconds:.1f}s is over {MAX_SECONDS}s, training would drop it")
    if seconds < 0.5:
        warnings.append("under half a second, did it record?")
    return warnings


def is_built_in(device_name: str) -> bool:
    return BUILT_IN_MIC.lower() in device_name.lower()


def save_take(path: Path, audio: np.ndarray, sr: int = SR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sr, subtype="PCM_24")


class SessionLog:
    """One JSON line per event in session_log.jsonl, so nothing is lost on a crash."""

    def __init__(self, path: Path, device: str):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.write(event="start", device=device, sample_rate=SR)

    def write(self, **fields) -> None:
        fields["time"] = datetime.now().isoformat(timespec="seconds")
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(fields, ensure_ascii=False) + "\n")


# ---------- audio sources ----------

class Mic:
    """Records from a real input device until stop() is called."""

    def __init__(self, device_name: str | None):
        import sounddevice as sd

        self.sd = sd
        self.index, self.name = self._pick(device_name)
        self.level = -120.0

    def _pick(self, wanted: str | None) -> tuple[int, str]:
        devices = self.sd.query_devices()
        inputs = [(i, d["name"]) for i, d in enumerate(devices) if d["max_input_channels"] > 0]
        if not inputs:
            sys.exit("No input device found.")
        target = wanted or BUILT_IN_MIC
        for i, name in inputs:
            if target.lower() in name.lower():
                return i, name
        default = self.sd.default.device[0]
        return default, devices[default]["name"]

    def record(self, stop: threading.Event) -> np.ndarray:
        chunks = []

        def callback(indata, frames, t, status):
            chunks.append(indata[:, 0].copy())
            self.level = peak_db(indata)

        with self.sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                                 device=self.index, callback=callback):
            stop.wait()
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype="float32")


class FakeSignal:
    """Stands in for the mic in --dry-run: a tone at a chosen loudness."""

    name = "fake signal (dry run)"

    def __init__(self):
        self.level = -120.0
        self.amplitude = 0.3
        self.seconds = 2.0

    def record(self, stop: threading.Event) -> np.ndarray:
        t = np.arange(int(self.seconds * SR)) / SR
        audio = (self.amplitude * np.sin(2 * np.pi * 220 * t)).astype("float32")
        audio = np.clip(audio, -1.0, 1.0)
        for start in range(0, len(audio), SR // 4):  # show the meter moving, like a real take
            self.level = peak_db(audio[start:start + SR // 4])
            print("\r  " + meter_bar(self.level), end="", flush=True)
            time.sleep(0.05)
        print()
        return audio


# ---------- interactive loop ----------

def take_with_meter(source) -> np.ndarray:
    """Record until Enter is pressed, showing a live level meter."""
    stop = threading.Event()
    result = {}
    worker = threading.Thread(target=lambda: result.setdefault("audio", source.record(stop)))
    worker.start()
    if isinstance(source, FakeSignal):
        worker.join()
        return result["audio"]

    def show_meter():
        while not stop.is_set():
            print("\r  " + meter_bar(source.level) + "   (Enter to stop)", end="", flush=True)
            time.sleep(0.1)

    threading.Thread(target=show_meter, daemon=True).start()
    input()
    stop.set()
    worker.join()
    print()
    return result["audio"]


def show(n: int, total: int, p: dict, roman: dict[str, str]) -> None:
    print("=" * 70)
    print(f"{n}/{total}  [{p['id']}]  lang={p['lang']}\n")
    if p["id"] in roman:
        print(f"    READ:  {roman[p['id']]}\n")
        print(f"    (saved as: {p['text']})\n")
    else:
        print(f"    {p['text']}\n")


def record_prompts(prompts, out_dir: Path, source, log: SessionLog, ask=input, roman=None) -> str:
    """Walk the prompts. Returns 'done' or 'quit'."""
    for n, p in enumerate(prompts, 1):
        redos = 0
        while True:
            show(n, len(prompts), p, roman or {})
            ask("Enter to start...")
            audio = take_with_meter(source)
            for w in check_take(audio):
                print(f"  ! {w}")
            print(f"  {len(audio) / SR:.1f}s")
            choice = ask("  Enter keep   r redo   s skip   q quit > ").strip().lower()
            if choice == "r":
                redos += 1
                continue
            if choice == "q":
                log.write(event="quit", at=p["id"])
                return "quit"
            if choice == "s":
                log.write(event="skip", id=p["id"], redos=redos)
            else:
                save_take(out_dir / f"{p['id']}.wav", audio)
                log.write(event="take", id=p["id"], seconds=round(len(audio) / SR, 2),
                          redos=redos, warnings=check_take(audio))
            break
    return "done"


# ---------- modes ----------

def run_session(args) -> None:
    order_path = Path(args.order)
    if not order_path.exists():
        sys.exit(f"{order_path} not found. Run recording/order.py first, or pass --order.")
    mic = Mic(args.device)
    print(f"Input device: {mic.name}")
    if not is_built_in(mic.name):
        print("  WARNING: this is not the built-in MacBook mic. Bluetooth mics sound like a phone call.")

    remaining = todo(read_tsv(order_path), recorded_ids(RAW))
    print(f"{len(remaining)} prompts left to record.")
    out_dir = RAW / f"session_{args.session}"
    log = SessionLog(out_dir / "session_log.jsonl", mic.name)
    status = record_prompts(remaining, out_dir, mic, log, roman=load_roman(ROMAN))
    log.write(event="end", status=status)
    print(f"Saved in {out_dir}. {len(recorded_ids(RAW))} prompts recorded in total.")


def run_reference(args) -> None:
    lines = read_tsv(ROOT / "recording" / "reference.txt")
    mic = Mic(args.device)
    print(f"Input device: {mic.name}")
    if not is_built_in(mic.name):
        print("  WARNING: this is not the built-in MacBook mic.")
    out_dir = RAW / "reference"
    prompts = [{"id": f"ref_{r['lang']}", "lang": r["lang"], "text": r["text"]} for r in lines]
    log = SessionLog(out_dir / "session_log.jsonl", mic.name)
    record_prompts(prompts, out_dir, mic, log, roman=load_roman(ROMAN))
    print(f"Reference clips saved in {out_dir}")


def run_dry(args) -> None:
    """Scripted run with a fake signal: meter, clipping warning, save and resume."""
    global RAW
    tmp = Path(tempfile.mkdtemp(prefix="record_dry_"))
    RAW = tmp
    prompts = [
        {"id": "p9001", "lang": "en", "text": "Dry run sentence one."},
        {"id": "p9002", "lang": "en", "text": "Dry run sentence two."},
        {"id": "p9003", "lang": "hi", "text": "ड्राई रन वाक्य तीन।"},
    ]
    fake = FakeSignal()
    out_dir = tmp / "session_1"
    log = SessionLog(out_dir / "session_log.jsonl", fake.name)

    # Take 1 clips, gets redone at a normal level and kept. Take 2 is quiet but kept.
    # Then quit before take 3.
    script = iter(["", "r", "", "", "", "", "", "q"])
    levels = iter([1.5, 0.3, 0.02, 0.3])

    def ask(prompt):
        answer = next(script)
        if prompt.startswith("Enter to start"):
            fake.amplitude = next(levels)
        print(f"{prompt}{answer!r}")
        return answer

    print(f"--- dry run in {tmp} ---")
    record_prompts(prompts, out_dir, fake, log, ask=ask)
    done = recorded_ids(tmp)
    print(f"\nsaved: {sorted(done)}")
    left = todo(prompts, done)
    print(f"resume would start at: {[p['id'] for p in left]}")
    assert done == {"p9001", "p9002"}, done
    assert [p["id"] for p in left] == ["p9003"]
    assert sf.info(out_dir / "p9001.wav").samplerate == SR
    print(open(out_dir / "session_log.jsonl", encoding="utf-8").read())
    print("DRY RUN OK")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--session", type=int, help="session number, files go to data/raw/session_<n>/")
    mode.add_argument("--reference", action="store_true", help="record the 3 reference clips")
    mode.add_argument("--dry-run", action="store_true", help="test with a fake signal, no mic needed")
    ap.add_argument("--order", default=str(ROOT / "recording" / "order.tsv"),
                    help="TSV with id, lang, text in recording order (default: recording/order.tsv)")
    ap.add_argument("--device", help="part of the input device name (default: the MacBook mic)")
    args = ap.parse_args()

    try:
        if args.dry_run:
            run_dry(args)
        elif args.reference:
            run_reference(args)
        else:
            run_session(args)
    except KeyboardInterrupt:
        print("\nStopped. Every kept take is already saved.")


if __name__ == "__main__":
    main()
