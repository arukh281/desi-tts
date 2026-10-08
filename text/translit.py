"""Roman-script Hinglish -> Devanagari, so Hinglish can be sent to XTTS as "hi".

The P1 probe showed pretrained XTTS reads Roman Hinglish badly through "en" but
reads the same sentence well in Devanagari through "hi". This is the automatic
front-end a real system could use (the Part 7 inference fix):

  1. text/normalize.py in "hinglish" mode: ₹, numbers, times -> Roman Hindi words,
     acronyms -> spaced capital letters ("E M I")
  2. spaced capital letters -> Devanagari letter names (ई एम आई)
  3. every other Roman word -> Devanagari with IndicXlit (AI4Bharat, MIT licence),
     via the CTranslate2 conversion Singla0009/all-indic-transliteration, top-1 per word

    to_devanagari("Aapka EMI ₹8,500 ka hai.") -> ("आपका ई एम आई आठ हज़ार ...", rewrites)

Needs: pip install ctranslate2 huggingface_hub (model ~45 MB, downloaded once).
"""
import os
import re
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from text.normalize import HI_DEV, HI_ROMAN, LETTERS_HI, normalize  # noqa: E402

MODEL_REPO = "Singla0009/all-indic-transliteration"
MODEL_DIR = "indicxlit_ct2_fp32"
# IndicXlit alone spells short words out as letters ("do" -> "ईडीओ", "aap" ->
# "आईएपी") and misses common ones ("kal" -> "काल"). So common words come from a
# small lexicon: the number words the normaliser itself produces (exact table),
# plus everyday Hinglish function words. IndicXlit handles everything else
# (names, places, English loanwords).
NUMBER_WORDS = dict(zip(HI_ROMAN, HI_DEV)) | {
    "zero": "ज़ीरो", "sau": "सौ", "hazaar": "हज़ार", "lakh": "लाख", "crore": "करोड़", "rupaye": "रुपये",
    "paise": "पैसे", "baje": "बजे", "saadhe": "साढ़े", "sava": "सवा", "paune": "पौने", "dedh": "डेढ़",
    "dhaai": "ढाई", "bajkar": "बजकर", "baj": "बज", "kar": "कर", "minute": "मिनट",
}
FUNCTION_WORDS = {
    "aap": "आप", "aapka": "आपका", "aapki": "आपकी", "aapke": "आपके", "aapko": "आपको", "aapse": "आपसे",
    "hum": "हम", "humne": "हमने", "hamari": "हमारी", "hamara": "हमारा", "main": "मैं", "mera": "मेरा",
    "meri": "मेरी", "hai": "है", "hain": "हैं", "ho": "हो", "tha": "था", "thi": "थी", "hoga": "होगा",
    "hogi": "होगी", "ka": "का", "ki": "की", "ke": "के", "ko": "को", "se": "से", "mein": "में",
    "par": "पर", "pe": "पे", "tak": "तक", "aur": "और", "ya": "या", "toh": "तो", "to": "तो",
    "kya": "क्या", "ji": "जी", "ek": "एक", "do": "दो", "kal": "कल", "aaj": "आज", "abhi": "अभी",
    "din": "दिन", "liye": "लिए", "band": "बंद", "mat": "मत", "ise": "इसे", "isi": "इसी", "agar": "अगर",
    "nahi": "नहीं", "nahin": "नहीं", "gaya": "गया", "gayi": "गई", "jayega": "जाएगा", "naam": "नाम",
    "karna": "करना", "karke": "करके", "kariye": "करिए", "dijiye": "दीजिए", "bata": "बता", "subah": "सुबह",
    "shaam": "शाम", "wali": "वाली", "wala": "वाला", "dobara": "दोबारा", "sakte": "सकते",
}
OVERRIDES: dict[str, str] = NUMBER_WORDS | FUNCTION_WORDS


@lru_cache(maxsize=1)
def _translator():
    import ctranslate2
    from huggingface_hub import snapshot_download

    path = snapshot_download(MODEL_REPO, allow_patterns=f"{MODEL_DIR}/*")
    return ctranslate2.Translator(os.path.join(path, MODEL_DIR), device="cpu")


def _xlit(words: list[str]) -> dict[str, str]:
    todo = sorted({w for w in words if w not in OVERRIDES})
    if not todo:
        return {}
    results = _translator().translate_batch([["__hi__"] + list(w) for w in todo], beam_size=4)
    return {w: "".join(r.hypotheses[0]) for w, r in zip(todo, results)}


def to_devanagari(text: str) -> tuple[str, list[tuple[str, str]]]:
    spoken, rewrites = normalize(text, "hinglish")
    tokens = re.findall(r"[A-Za-z]+|[^A-Za-z]+", spoken)
    words = [t.lower() for t in tokens if t.isalpha() and not (len(t) == 1 and t.isupper())]
    found = _xlit(words)
    out = []
    for t in tokens:
        if not t.isalpha():
            out.append(t.replace(".", "।") if t.strip() == "." or t.endswith(".") else t)
        elif len(t) == 1 and t.isupper():
            out.append(LETTERS_HI[t])
        else:
            out.append(OVERRIDES.get(t.lower(), found.get(t.lower(), t)))
    return "".join(out), rewrites
