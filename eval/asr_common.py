"""Shared Whisper + scoring helpers for eval/asr_eval.py and data_prep/asr_check.py.

Model: openai/whisper-large-v3, run through transformers in fp16. Why this one:
it's the strongest Whisper for Hindi, our hardest language; the turbo variant
cuts the decoder from 32 layers to 4 and loses multilingual accuracy; and at
about 3 GB in fp16 it fits easily on a 16 GB T4.

The language is forced per row (en for English and Hinglish, hi for Hindi)
rather than auto-detected, so a mis-detected language can't masquerade as
a TTS error.

Scoring: both texts go through text/normalize.py (so "₹50" and "fifty rupees"
compare equal), then lowercase, punctuation stripped, Devanagari nukta and
chandrabindu folded, spaces collapsed. CER and WER are plain edit distances.
For Hindi, Whisper writes English loanwords in Latin script ("आपकी booking"),
so Latin words are first transliterated to Devanagari (text/translit.py,
IndicXlit); without that, correct speech scores as wrong.
"""
import re
import string
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from text.normalize import normalize  # noqa: E402

WHISPER_MODEL = "openai/whisper-large-v3"
WHISPER_LANG = {"en": "en", "hinglish": "en", "hi": "hi"}
PUNCT = string.punctuation + "।॥“”‘’…–—"


def latin_to_devanagari(text: str) -> str:
    """Latin-script words inside Hindi text -> Devanagari. No-op if IndicXlit isn't installed."""
    if not re.search(r"[A-Za-z]{2,}", text):
        return text
    try:
        from text.translit import to_devanagari
    except ImportError:
        return text
    parts = re.split(r"([A-Za-z][A-Za-z ]*[A-Za-z]|[A-Za-z])", text)
    return "".join(to_devanagari(p)[0].rstrip("।") if re.fullmatch(r"[A-Za-z ]+", p) else p for p in parts)


def score_text(text: str, lang: str) -> str:
    if lang == "hi":
        text = latin_to_devanagari(text)
    text = normalize(text, lang)[0]
    text = unicodedata.normalize("NFD", text).replace("़", "")  # nukta: ज़ -> ज
    text = unicodedata.normalize("NFC", text).replace("ँ", "ं")  # chandrabindu -> anusvara
    text = text.lower().translate(str.maketrans(PUNCT, " " * len(PUNCT)))
    return re.sub(r"\s+", " ", text).strip()


def edit_distance(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    ref, hyp = ref.replace(" ", ""), hyp.replace(" ", "")
    return edit_distance(list(ref), list(hyp)) / max(1, len(ref))


def wer(ref: str, hyp: str) -> float:
    return edit_distance(ref.split(), hyp.split()) / max(1, len(ref.split()))


def load_whisper(model: str = WHISPER_MODEL):
    import torch
    from transformers import pipeline

    return pipeline("automatic-speech-recognition", model=model, torch_dtype=torch.float16, device=0)


def transcribe(asr, paths: list[str], langs: list[str], batch_size: int = 8) -> list[str]:
    """Transcribe in batches, one forced language per batch."""
    out = {}
    for lang in sorted(set(langs)):
        group = [p for p, l in zip(paths, langs) if l == lang]
        results = asr(group, batch_size=batch_size,
                      generate_kwargs={"language": WHISPER_LANG[lang], "task": "transcribe"})
        out.update({p: r["text"].strip() for p, r in zip(group, results)})
    return [out[p] for p in paths]
