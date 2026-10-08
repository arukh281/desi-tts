"""Rule-based text normaliser: written form -> the spoken form our data uses.

Infrastructure for the spoken-form convention in recording/prompts.tsv (numbers
and amounts as words, acronyms as spaced letters), used by inference
(infer/synthesize.py --normalize on) and by evaluation. It is not the Part 7 fix.

    normalize("Your EMI of ₹12,750 is due.", "en")
    -> ("Your E M I of twelve thousand seven hundred and fifty rupees is due.",
        [("₹12,750", "twelve thousand seven hundred and fifty rupees"), ("EMI", "E M I")])

lang is "en", "hinglish" or "hi". Deterministic: same input, same output.
Rules run in a fixed order (currency, dates, times, digit strings, decimals,
ordinals, numbers, acronyms), each only on text the earlier ones didn't touch.

Conventions (from prompts.tsv; see NOTES.md for the choices without a precedent):
  English   "hundred and fifty", Indian grouping (lakh, crore), "the fifth of
            November", "ten thirty in the morning", digit strings digit by digit
  Hinglish  Hindi number words in Roman script ("paanch hazaar rupaye", "das baje")
  Hindi     Devanagari number words, साढ़े / सवा / पौने for times, Latin acronyms
            as Devanagari letter names (OTP -> ओ टी पी)
"""
import re

# ---------- number words ----------

EN_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
           "fifteen sixteen seventeen eighteen nineteen").split()
EN_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
EN_ORD = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
          "nine": "ninth", "twelve": "twelfth"}

HI_ROMAN = """ek do teen chaar paanch chhe saat aath nau das gyaarah barah terah chaudah pandrah solah
satrah atharah unnees bees ikkees baees teis chaubees pachchis chhabbees sattaais atthaais untees tees
iktees battees taintees chauntees paintis chhattees saintees adtees untaalees chaalis iktaalees bayaalees
taintaalees chavaalees paintaalees chhiyaalees saintaalees adtaalees unchaas pachaas ikyaavan baavan
tirpan chauvan pachpan chhappan sattaavan atthaavan unsath saath iksath baasath tirsath chaunsath
painsath chhiyaasath sadsath adsath unhattar sattar ikhattar bahattar tihattar chauhattar pachhattar
chhihattar sathattar athhattar unyaasi assi ikyaasi bayaasi tiraasi chauraasi pachaasi chhiyaasi
sattaasi atthaasi navaasi nabbe ikyaanve baanve tiraanve chauraanve pachaanve chhiyaanve sattaanve
atthaanve ninyaanve""".split()
HI_DEV = """एक दो तीन चार पाँच छह सात आठ नौ दस ग्यारह बारह तेरह चौदह पंद्रह सोलह सत्रह अठारह उन्नीस
बीस इक्कीस बाईस तेईस चौबीस पच्चीस छब्बीस सत्ताईस अट्ठाईस उनतीस तीस इकतीस बत्तीस तैंतीस चौंतीस
पैंतीस छत्तीस सैंतीस अड़तीस उनतालीस चालीस इकतालीस बयालीस तैंतालीस चवालीस पैंतालीस छियालीस
सैंतालीस अड़तालीस उनचास पचास इक्यावन बावन तिरपन चौवन पचपन छप्पन सत्तावन अट्ठावन उनसठ साठ इकसठ
बासठ तिरसठ चौंसठ पैंसठ छियासठ सड़सठ अड़सठ उनहत्तर सत्तर इकहत्तर बहत्तर तिहत्तर चौहत्तर पचहत्तर
छिहत्तर सतहत्तर अठहत्तर उन्यासी अस्सी इक्यासी बयासी तिरासी चौरासी पचासी छियासी सत्तासी अट्ठासी
नवासी नब्बे इक्यानवे बानवे तिरानवे चौरानवे पंचानवे छियानवे सत्तानवे अट्ठानवे निन्यानवे""".split()
assert len(HI_ROMAN) == len(HI_DEV) == 99

HI_WORDS = {
    "hinglish": {"zero": "zero", "sau": "sau", "hazaar": "hazaar", "lakh": "lakh", "crore": "crore",
                 "rupees": "rupaye", "paise": "paise", "and": "aur", "point": "point"},
    "hi": {"zero": "ज़ीरो", "sau": "सौ", "hazaar": "हज़ार", "lakh": "लाख", "crore": "करोड़",
           "rupees": "रुपये", "paise": "पैसे", "and": "और", "point": "दशमलव"},
}

MONTHS_EN = ("January February March April May June July August September October November "
             "December").split()
MONTHS_HI = "जनवरी फ़रवरी मार्च अप्रैल मई जून जुलाई अगस्त सितंबर अक्टूबर नवंबर दिसंबर".split()
MONTH_ABBR = {m[:3].lower(): i for i, m in enumerate(MONTHS_EN, 1)}

LETTERS_HI = dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                      "ए बी सी डी ई एफ़ जी एच आई जे के एल एम एन ओ पी क्यू आर एस टी यू वी डब्ल्यू एक्स वाई ज़ेड".split()))

# Acronyms said as words rather than letters. Empty on purpose: prompts.tsv spells
# PAN and PIN as letters ("P A N card", "card P I N"), and inference must match the
# training data. Add words here only if the recordings change.
WORD_ACRONYMS: set[str] = set()


def en_under_1000(n: int) -> str:
    if n < 20:
        return EN_ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return EN_TENS[tens] + (f" {EN_ONES[ones]}" if ones else "")
    hundreds, rest = divmod(n, 100)
    return f"{EN_ONES[hundreds]} hundred" + (f" and {en_under_1000(rest)}" if rest else "")


def hi_under_100(n: int, lang: str) -> str:
    if n == 0:
        return HI_WORDS[lang]["zero"]
    return (HI_ROMAN if lang == "hinglish" else HI_DEV)[n - 1]


def number_words(n: int, lang: str) -> str:
    """Indian grouping: crore, lakh, thousand, hundred."""
    if n == 0:
        return "zero" if lang == "en" else HI_WORDS[lang]["zero"]
    parts = []
    for size, name in ((10**7, "crore"), (10**5, "lakh"), (1000, "thousand" if lang == "en" else "hazaar")):
        count, n = divmod(n, size)
        if count:
            label = name if lang == "en" else HI_WORDS[lang][name]
            parts.append(f"{number_words(count, lang)} {label}")
    if n:
        if lang == "en":
            parts.append(en_under_1000(n))
        else:
            hundreds, rest = divmod(n, 100)
            if hundreds:
                parts.append(f"{hi_under_100(hundreds, lang)} {HI_WORDS[lang]['sau']}")
            if rest:
                parts.append(hi_under_100(rest, lang))
    return " ".join(parts)


def digit_words(digits: str, lang: str) -> str:
    if lang == "en":
        return " ".join(EN_ONES[int(d)] for d in digits)
    return " ".join(hi_under_100(int(d), lang) for d in digits)


def ordinal_en(n: int) -> str:
    words = number_words(n, "en").split()
    last = words[-1]
    if last in EN_ORD:
        words[-1] = EN_ORD[last]
    elif last.endswith("y"):
        words[-1] = last[:-1] + "ieth"
    else:
        words[-1] = last + "th"
    return " ".join(words)


def year_words(year: int, lang: str) -> str:
    if lang != "en":
        return number_words(year, lang)
    if 2000 <= year <= 2009:
        return number_words(year, "en")
    high, low = divmod(year, 100)
    return f"{number_words(high, 'en')} {number_words(low, 'en') if low else 'hundred'}"


def decimal_words(text: str, lang: str) -> str:
    whole, frac = text.split(".")
    return f"{number_words(int(whole), lang)} {HI_WORDS.get(lang, {}).get('point', 'point')} {digit_words(frac, lang)}"


# ---------- rules ----------

AMOUNT = r"(\d{1,3}(?:,\d{2,3})+|\d+)(?:\.(\d{1,2}))?"
CURRENCY = re.compile(r"(?:₹\s?|\bRs\.?\s?|\bINR\s?)" + AMOUNT + r"(?:\s?(lakh|crore)\b)?")
DATE_NUM = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")
DATE_ORD = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(" + "|".join(MONTHS_EN) + r"|"
                      + "|".join(m[:3] for m in MONTHS_EN) + r")\b(?:\s+(\d{4}))?")
TIME_12 = re.compile(r"\b(\d{1,2})(?:[:.](\d{2}))?\s?(AM|PM|am|pm|a\.m\.|p\.m\.)")
# Whisper writes times as "10.30 in the morning" and amounts as "5000 rupees".
TIME_DOT = re.compile(r"\b(\d{1,2})\.(\d{2})\b(?=\s+(?:in the|this|tonight|at night|baje|बजे))")
# Python's \b doesn't see the end of Devanagari words ending in a vowel sign, so look ahead instead.
AMOUNT_WORD = re.compile(r"\b" + AMOUNT + r"\s?(rupees|rupee|rupaye|rupay|रुपये|रुपए|रुपे)(?=[\s.,!?।]|$)", re.I)
GROUPED = re.compile(r"\b\d{1,3}(?:,\d{2,3})+\b")
TIME_24 = re.compile(r"\b(\d{1,2}):(\d{2})\b")
BAJE = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s?(baje|बजे)")
LONG_DIGITS = re.compile(r"\b\d{4,}\b")
CALL_CONTEXT = re.compile(r"\b(\d{2,3})\b(?=\s+(?:pe|par|par)\s+call)|(?<=\bcall\s)(\d{2,3})\b|(?<=\bdial\s)(\d{2,3})\b")
DECIMAL = re.compile(r"\b\d+\.\d+\b")
ORDINAL = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b")
NUMBER = re.compile(r"\b\d+\b")
ACRONYM = re.compile(r"\b[A-Z]{2,5}\b")
PERCENT = re.compile(r"\b(\d+(?:\.\d+)?)\s?%")


def say_amount(whole: str, frac: str | None, scale: str | None, lang: str) -> str:
    w = HI_WORDS.get(lang, {})
    rupees = w.get("rupees", "rupees")
    if scale:  # ₹2.5 lakh -> two point five lakh rupees
        number = decimal_words(f"{whole}.{frac}", lang) if frac else number_words(int(whole), lang)
        label = scale if lang == "en" else w[scale]
        return f"{number} {label} {rupees}"
    out = f"{number_words(int(whole.replace(',', '')), lang)} {rupees}"
    if frac and int(frac):
        paise = int(frac.ljust(2, "0"))
        out += f" {w.get('and', 'and')} {number_words(paise, lang)} {w.get('paise', 'paise')}"
    return out


def day_period(hour24: int) -> str:
    if hour24 < 4:
        return "at night"
    if hour24 < 12:
        return "in the morning"
    if hour24 < 17:
        return "in the afternoon"
    if hour24 < 21:
        return "in the evening"
    return "at night"


def clock_en(hour: int, minute: int) -> str:
    return number_words(hour, "en") + (f" {number_words(minute, 'en')}" if minute else "")


def clock_hindi(hour: int, minute: int, lang: str) -> str:
    """साढ़े दस, सवा नौ, पौने आठ, डेढ़, ढाई; other minutes as 'X baj kar Y minute'."""
    h = hour % 12 or 12
    nxt = h % 12 + 1
    say = lambda n: hi_under_100(n, lang)
    words = {"hinglish": {"half": "saadhe", "quarter": "sava", "less": "paune", 1.5: "dedh", 2.5: "dhaai",
                          "past": "baj kar", "min": "minute"},
             "hi": {"half": "साढ़े", "quarter": "सवा", "less": "पौने", 1.5: "डेढ़", 2.5: "ढाई",
                    "past": "बजकर", "min": "मिनट"}}[lang]
    if minute == 0:
        return say(h)
    if minute == 30:
        return words[1.5] if h == 1 else words[2.5] if h == 2 else f"{words['half']} {say(h)}"
    if minute == 15:
        return f"{words['quarter']} {say(h)}"
    if minute == 45:
        return f"{words['less']} {say(nxt)}"
    return f"{say(h)} {words['past']} {say(minute)} {words['min']}"


def normalize(text: str, lang: str = "en") -> tuple[str, list[tuple[str, str]]]:
    if lang not in ("en", "hinglish", "hi"):
        raise ValueError(f"lang must be en, hinglish or hi, not {lang!r}")
    rewrites: list[tuple[str, str]] = []

    def sub(pattern: re.Pattern, fn) -> None:
        nonlocal text

        def repl(m: re.Match) -> str:
            spoken = fn(m)
            if spoken is None:
                return m.group(0)
            rewrites.append((m.group(0), spoken))
            return spoken

        text = pattern.sub(repl, text)

    sub(CURRENCY, lambda m: say_amount(m.group(1), m.group(2), m.group(3), lang))
    sub(AMOUNT_WORD, lambda m: say_amount(m.group(1), m.group(2), None, lang))
    sub(GROUPED, lambda m: number_words(int(m.group(0).replace(",", "")), lang))
    sub(PERCENT, lambda m: (decimal_words(m.group(1), lang) if "." in m.group(1)
                            else number_words(int(m.group(1)), lang))
        + (" per cent" if lang == "en" else " pratishat" if lang == "hinglish" else " प्रतिशत"))

    def numeric_date(m: re.Match) -> str:
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if lang == "en":
            return f"the {ordinal_en(day)} of {MONTHS_EN[month - 1]} {year_words(year, 'en')}"
        months = MONTHS_EN if lang == "hinglish" else MONTHS_HI
        return f"{number_words(day, lang)} {months[month - 1]} {year_words(year, lang)}"

    sub(DATE_NUM, numeric_date)

    def word_date(m: re.Match) -> str:
        day, month = int(m.group(1)), MONTH_ABBR[m.group(2)[:3].lower()]
        year = f" {year_words(int(m.group(3)), lang)}" if m.group(3) else ""
        if lang == "en":
            return f"the {ordinal_en(day)} of {MONTHS_EN[month - 1]}{year}"
        return f"{number_words(day, lang)} {MONTHS_EN[month - 1]}{year}"

    sub(DATE_ORD, word_date)

    def time_12(m: re.Match) -> str:
        hour, minute = int(m.group(1)), int(m.group(2) or 0)
        pm = m.group(3).lower().startswith("p")
        hour24 = hour % 12 + (12 if pm else 0)
        return f"{clock_en(hour, minute)} {day_period(hour24)}"

    sub(TIME_12, time_12)
    sub(TIME_DOT, lambda m: clock_en(int(m.group(1)), int(m.group(2))) if lang == "en"
        else clock_hindi(int(m.group(1)), int(m.group(2)), lang))
    sub(BAJE, lambda m: f"{clock_hindi(int(m.group(1)), int(m.group(2) or 0), 'hi' if lang == 'hi' else 'hinglish')} {m.group(3)}")
    sub(TIME_24, lambda m: clock_en(int(m.group(1)), int(m.group(2))) if lang == "en"
        else clock_hindi(int(m.group(1)), int(m.group(2)), lang))
    sub(LONG_DIGITS, lambda m: digit_words(m.group(0), lang))
    sub(CALL_CONTEXT, lambda m: digit_words(next(g for g in m.groups() if g), lang))
    sub(DECIMAL, lambda m: decimal_words(m.group(0), lang))
    sub(ORDINAL, lambda m: ordinal_en(int(m.group(1))) if lang == "en" else number_words(int(m.group(1)), lang))
    sub(NUMBER, lambda m: number_words(int(m.group(0)), lang))

    def acronym(m: re.Match) -> str | None:
        word = m.group(0)
        if word in WORD_ACRONYMS:
            return None
        return " ".join(LETTERS_HI[c] for c in word) if lang == "hi" else " ".join(word)

    sub(ACRONYM, acronym)
    return text, rewrites
