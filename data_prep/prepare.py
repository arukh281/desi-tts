"""Step 1 of data prep (runs on the Mac, no ML): check and clean every recorded clip.

For each data/raw/session_*/<id>.wav (48 kHz masters, never modified) it:
  1. measures sample rate, duration, peak, RMS, clipped samples, noise floor, SNR
  2. mutes Enter-key clicks in the first/last ~200 ms (short bursts cut off from the speech)
  3. trims leading/trailing silence, keeping a small pad
  4. measures the longest pause left inside the speech
  5. loudness-normalises (LUFS) and resamples to 22.05 kHz mono for training

Writes data/processed/wavs/<id>.wav and data/processed/clips.csv with every
measurement plus a status (keep / flag / drop) and the reasons. The Whisper
transcript check comes later (asr_check.py) and finalize.py combines both.

Usage:
  python data_prep/prepare.py
  python data_prep/prepare.py --raw data/raw --prompts recording/prompts.tsv --out data/processed
"""
import argparse
import csv
import json
from math import gcd
from pathlib import Path

import numpy as np
import pyloudnorm
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "data_prep" / "config.json"


def load_config(path: Path = CONFIG) -> dict:
    return {k: v for k, v in json.load(open(path)).items() if not k.startswith("_")}


def db(x: float) -> float:
    return 20 * np.log10(max(x, 1e-9))


def frame_rms_db(audio: np.ndarray, sr: int, frame_ms: int) -> np.ndarray:
    n = max(1, int(sr * frame_ms / 1000))
    frames = audio[: len(audio) // n * n].reshape(-1, n)
    return 20 * np.log10(np.sqrt((frames**2).mean(axis=1)) + 1e-9)


def noise_and_snr(frames_db: np.ndarray, edge_frames: int) -> tuple[float, float]:
    """Noise floor and SNR, in dB.

    Takes start and end in silence (I press Enter, then speak), so the edges
    are the best sample of the room. Quiet dips inside continuous speech would
    otherwise pass for noise. Floor = the lower of the quietest 10% of the
    whole take and the quietest 20% of the edge frames. Speech = 95th percentile.
    """
    floor = float(np.percentile(frames_db, 10))
    if len(frames_db) > 2 * edge_frames:
        edges = np.concatenate([frames_db[:edge_frames], frames_db[-edge_frames:]])
        floor = min(floor, float(np.percentile(edges, 20)))
    speech = float(np.percentile(frames_db, 95))
    return floor, speech - floor


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) index pairs of consecutive True values, end exclusive."""
    out, start = [], None
    for i, on in enumerate(mask):
        if on and start is None:
            start = i
        elif not on and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def remove_edge_clicks(audio, sr, active, cfg) -> tuple[np.ndarray, int]:
    """Mute short bursts near the edges that are separate from the main speech.

    The Enter key is pressed on the MacBook right next to the mic, so takes
    often start or end with a click. A click is an active run shorter than
    click_max_ms inside the first/last edge_window_s, with silence between it
    and the speech.
    """
    frame = int(sr * cfg["frame_ms"] / 1000)
    edge = int(cfg["edge_window_s"] * 1000 / cfg["frame_ms"])
    max_len = max(1, cfg["click_max_ms"] // cfg["frame_ms"])
    segments = runs(active)
    if len(segments) < 2:
        return audio, 0
    longest = max(segments, key=lambda s: s[1] - s[0])
    out, removed = audio.copy(), 0
    for start, end in segments:
        if (start, end) == longest or end - start > max_len:
            continue
        near_start = end <= edge and end < longest[0]
        near_end = start >= len(active) - edge and start > longest[1]
        if near_start or near_end:
            out[start * frame:end * frame] = 0.0
            active[start:end] = False
            removed += 1
    return out, removed


def trim(audio, sr, active, cfg) -> tuple[np.ndarray, float]:
    """Cut to first..last active frame plus a pad. Returns audio and the longest inner pause."""
    frame = int(sr * cfg["frame_ms"] / 1000)
    idx = np.flatnonzero(active)
    if idx.size == 0:
        return audio[:0], 0.0
    pad = int(cfg["trim_pad_s"] * sr)
    start = max(0, idx[0] * frame - pad)
    end = min(len(audio), (idx[-1] + 1) * frame + pad)
    inner = active[idx[0]:idx[-1] + 1]
    gaps = [e - s for s, e in runs(~inner)]
    longest_pause = max(gaps, default=0) * cfg["frame_ms"] / 1000
    return audio[start:end], longest_pause


def normalise_loudness(audio, sr, cfg) -> np.ndarray:
    meter = pyloudnorm.Meter(sr)
    loud = meter.integrated_loudness(audio)
    if not np.isfinite(loud):
        return audio
    out = pyloudnorm.normalize.loudness(audio, loud, cfg["loudness_lufs"])
    ceiling = 10 ** (cfg["peak_ceiling_dbfs"] / 20)
    peak = np.abs(out).max()
    return out * (ceiling / peak) if peak > ceiling else out


def resample(audio, sr_in, sr_out) -> np.ndarray:
    g = gcd(sr_in, sr_out)
    return resample_poly(audio, sr_out // g, sr_in // g).astype("float32")


def process_clip(path: Path, cfg: dict) -> tuple[dict, np.ndarray | None]:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    stats = {"sample_rate": sr, "raw_seconds": round(len(audio) / sr, 2)}
    stats["peak_dbfs"] = round(db(float(np.abs(audio).max(initial=0.0))), 1)
    stats["rms_dbfs"] = round(db(float(np.sqrt((audio**2).mean()))) if audio.size else -180.0, 1)
    stats["clipped_samples"] = int((np.abs(audio) >= cfg["clip_level"]).sum())

    frames = frame_rms_db(audio, sr, cfg["frame_ms"])
    floor, snr = noise_and_snr(frames, int(cfg["edge_window_s"] * 1000 / cfg["frame_ms"]))
    stats["noise_floor_dbfs"] = round(floor, 1)
    stats["snr_db"] = round(snr, 1)

    active = frames > floor + cfg["active_above_floor_db"]
    audio, clicks = remove_edge_clicks(audio, sr, active, cfg)
    stats["edge_clicks_removed"] = clicks
    audio, pause = trim(audio, sr, active, cfg)
    stats["seconds"] = round(len(audio) / sr, 2)
    stats["longest_pause_s"] = round(pause, 2)

    if len(audio) < sr * 0.3:
        return stats, None
    out = resample(normalise_loudness(audio, sr, cfg), sr, cfg["train_sample_rate"])
    return stats, out


def judge(stats: dict, cfg: dict) -> tuple[str, list[str]]:
    """keep / flag / drop, with a reason for every flag and drop."""
    drop, flag = [], []
    if stats["sample_rate"] != cfg["master_sample_rate"]:
        flag.append(f"sample_rate {stats['sample_rate']}")
    if stats["seconds"] > cfg["max_seconds"]:
        drop.append(f"too_long {stats['seconds']}s")
    if stats["seconds"] == 0:
        drop.append("no_speech_found")
    elif stats["seconds"] < cfg["min_seconds"]:
        drop.append(f"too_short {stats['seconds']}s")
    if stats["clipped_samples"] >= cfg["clip_drop_samples"]:
        drop.append(f"clipping {stats['clipped_samples']} samples")
    elif stats["clipped_samples"] >= cfg["clip_flag_samples"]:
        flag.append(f"clipping {stats['clipped_samples']} samples")
    if stats["peak_dbfs"] < cfg["min_peak_dbfs"]:
        drop.append(f"too_quiet peak {stats['peak_dbfs']} dBFS")
    if stats["snr_db"] < cfg["snr_drop_db"]:
        drop.append(f"low_snr {stats['snr_db']} dB")
    elif stats["snr_db"] < cfg["snr_flag_db"]:
        flag.append(f"low_snr {stats['snr_db']} dB")
    if stats["longest_pause_s"] > cfg["pause_drop_s"]:
        drop.append(f"long_pause {stats['longest_pause_s']}s")
    elif stats["longest_pause_s"] > cfg["pause_flag_s"]:
        flag.append(f"long_pause {stats['longest_pause_s']}s")
    if stats["edge_clicks_removed"]:
        flag.append(f"edge_clicks_removed {stats['edge_clicks_removed']}")
    status = "drop" if drop else "flag" if flag else "keep"
    return status, drop + flag


def take_order(session_dir: Path) -> dict[str, int]:
    """Order clips were kept in, from session_log.jsonl (falls back to file time)."""
    log = session_dir / "session_log.jsonl"
    order = {}
    if log.exists():
        for line in open(log, encoding="utf-8"):
            event = json.loads(line)
            if event.get("event") == "take":
                order[event["id"]] = len(order)
    for wav in sorted(session_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime):
        order.setdefault(wav.stem, len(order))
    return order


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--prompts", default=str(ROOT / "recording" / "prompts.tsv"))
    ap.add_argument("--out", default=str(ROOT / "data" / "processed"))
    ap.add_argument("--config", default=str(CONFIG))
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    with open(args.prompts, encoding="utf-8", newline="") as f:
        prompts = {r["id"]: r for r in csv.DictReader(f, delimiter="\t")}
    out = Path(args.out)
    (out / "wavs").mkdir(parents=True, exist_ok=True)

    rows = []
    for session_dir in sorted(Path(args.raw).glob("session_*")):
        order = take_order(session_dir)
        for wav in sorted(session_dir.glob("*.wav")):
            prompt = prompts.get(wav.stem)
            if prompt is None:
                print(f"skip {wav}: id not in prompts")
                continue
            stats, audio = process_clip(wav, cfg)
            status, reasons = judge(stats, cfg)
            if audio is not None:
                sf.write(out / "wavs" / f"{wav.stem}.wav", audio, cfg["train_sample_rate"], subtype="PCM_16")
            rows.append({"id": wav.stem, "session": session_dir.name, "take_order": order.get(wav.stem, -1),
                         "lang": prompt["lang"], "text": prompt["text"], **stats,
                         "status": status, "reasons": "; ".join(reasons)})

    if not rows:
        raise SystemExit(f"no clips found under {args.raw}/session_*/")
    with open(out / "clips.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = {s: sum(r["status"] == s for r in rows) for s in ("keep", "flag", "drop")}
    print(f"{len(rows)} clips -> {out / 'clips.csv'}  {counts}")


if __name__ == "__main__":
    main()
