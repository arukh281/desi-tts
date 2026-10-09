"""Speaker consistency: how close each synthesised clip sounds to the reference voice.

Uses SpeechBrain's ECAPA-TDNN speaker encoder (speechbrain/spkrec-ecapa-voxceleb),
which is independent of XTTS. XTTS's own speaker encoder would grade its own
homework. Each clip's embedding is compared by cosine similarity to the mean
embedding of the reference clip(s). 1.0 = same voice print; unrelated voices
usually land well below 0.5.

If the reference files are named ref_en / ref_hinglish / ref_hi, each row is
compared with the reference in its own language: the same voice scores much
lower across languages (seen in the P1 probe), so mixing them would mislead.
Otherwise the mean of all references is used.

Writes <synth-dir>/speaker_sim.csv and prints mean/min per language.

Usage (GPU or CPU):
  python eval/speaker_sim.py --synth-dir out/synth --ref data/raw/reference/*.wav
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

ENCODER = "speechbrain/spkrec-ecapa-voxceleb"
SR = 16_000


def embed(encoder, path: str):
    import torch
    import torchaudio

    wav, sr = torchaudio.load(path)
    wav = torchaudio.functional.resample(wav.mean(dim=0, keepdim=True), sr, SR)
    with torch.no_grad():
        return encoder.encode_batch(wav).squeeze()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synth-dir", required=True)
    ap.add_argument("--ref", nargs="+", required=True, help="reference wav(s) of the target voice")
    args = ap.parse_args()

    import torch
    from speechbrain.inference.speaker import EncoderClassifier

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = EncoderClassifier.from_hparams(source=ENCODER, savedir="/tmp/ecapa", run_opts={"device": device})
    refs = {Path(p).stem.replace("ref_", ""): embed(encoder, p) for p in args.ref}
    mean_ref = torch.stack(list(refs.values())).mean(dim=0)

    synth = Path(args.synth_dir)
    rows = [r for r in csv.DictReader(open(synth / "manifest.csv", encoding="utf-8")) if not r.get("error")]
    out, by_lang = [], defaultdict(list)
    for row in rows:
        ref = refs.get(row["lang"], mean_ref)
        sim = torch.nn.functional.cosine_similarity(embed(encoder, str(synth / f"{row['id']}.wav")), ref, dim=0)
        out.append({"id": row["id"], "lang": row["lang"], "cosine": round(float(sim), 3)})
        by_lang[row["lang"]].append(float(sim))
        print(f"{row['id']} {row['lang']:8} {float(sim):.3f}")

    with open(synth / "speaker_sim.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "lang", "cosine"])
        writer.writeheader()
        writer.writerows(out)
    for lang, sims in sorted(by_lang.items()):
        print(f"lang={lang:8} n={len(sims)} mean {sum(sims) / len(sims):.3f} min {min(sims):.3f}")


if __name__ == "__main__":
    main()
