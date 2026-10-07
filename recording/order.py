"""Decide the order to record prompts in, hardest first.

Each prompt gets tags for the hard things it contains (names, places, numbers,
currency, acronyms, dates_times, hinglish, hindi). Prompts with more tags come
first, and the three language groups (English, Hinglish, Hindi) are interleaved
in proportion, so if only session 1 gets recorded it is still a balanced, useful
small dataset.

It also flags prompts likely to be too long for training. The XTTS GPT recipe
drops clips over max_wav_length = 255995 samples (11.6 s at 22.05 kHz) and text
over max_text_length = 200 characters (coqui-tts 0.27.5 defaults).

Usage:
  python recording/order.py                       writes recording/order.tsv
  python recording/order.py --prompts X --out Y
"""
import argparse
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MAX_SECONDS = 11.6
MAX_TEXT_CHARS = 200
CHARS_PER_SECOND = 14.0  # rough reading speed incl. spaces; only used to flag, not to cut

NUMBER_WORDS = (r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
                r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|"
                r"forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|lakh|lakhs|crore|"
                r"crores|ek|do|teen|char|paanch|hazaar|hazar|sau)\b")
HINDI_NUMBERS = "एक|दो|तीन|चार|पांच|पाँच|हज़ार|हजार|लाख|करोड़|सौ"
CURRENCY = r"\b(rupee|rupees|rupaye|rupay|paise)\b|रुपये|रुपए"
ACRONYM = r"\b[A-Z](?: [A-Z]){1,}\b"  # spelled-out acronyms like "K Y C"
DATES_TIMES = (r"\b(january|february|march|april|may|june|july|august|september|october|"
               r"november|december|monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
               r"today|tomorrow|yesterday|morning|evening|afternoon|night|o'clock|baje|kal|"
               r"week|month)\b|बजे|कल|सोमवार|मंगलवार|बुधवार|गुरुवार|शुक्रवार|शनिवार|रविवार")
HINGLISH_WORDS = (r"\b(hai|hain|hoga|hogi|aap|aapka|aapki|aapke|aapko|kar|karna|karein|karega|karegi|"
                  r"ho|gaya|gayi|jayega|jayegi|milega|milegi|nahi|kya|kab|kitna|kitne|mein|hum|humne|"
                  r"hoon|ke|ki|ka|ko|se|bhi|toh|aur|lekin|abhi|ji|haan|theek|bilkul|accha|achha|"
                  r"kripya|dhanyavaad|raha|rahi|rahe|diya|liya|wala|wali|uthaiyega)\b")
PLACES = {
    "Bengaluru", "Bangalore", "Mumbai", "Delhi", "Pune", "Chennai", "Hyderabad", "Kolkata",
    "Guwahati", "Jaipur", "Lucknow", "Ahmedabad", "Kochi", "Indiranagar", "Koramangala",
    "Thiruvananthapuram", "Noida", "Gurugram", "Gurgaon", "Chandigarh", "Bhopal", "Indore",
    "Nagpur", "Patna", "Mysuru", "Mysore", "Goa", "Surat", "Varanasi", "Whitefield", "Andheri",
}
NOT_NAMES = {"Mr", "Mrs", "Ms", "Dr", "I", "OK", "Okay", "Sir", "Madam", "Namaste", "Aapka",
             "Aapki", "Aap", "Please", "Thank", "Thanks", "Monday", "Tuesday", "Wednesday",
             "Thursday", "Friday", "Saturday", "Sunday", "January", "February", "March", "April",
             "May", "June", "July", "August", "September", "October", "November", "December"}

TAG_ORDER = ["names", "places", "numbers", "currency", "acronyms", "dates_times", "hinglish", "hindi"]


def language_group(row: dict) -> str:
    if row["lang"] == "hi":
        return "hi"
    return "hinglish" if re.search(HINGLISH_WORDS, row["text"], re.I) else "en"


def capitalised_inner_words(text: str) -> list[str]:
    tokens = text.split()
    out = []
    for i, tok in enumerate(tokens):
        word = tok.strip(".,!?;:'\"()")
        if not word or not word[0].isupper() or len(word) == 1:
            continue
        if i == 0 or tokens[i - 1][-1] in ".?!":
            continue
        out.append(word)
    return out


def tags_for(row: dict) -> list[str]:
    text = row["text"]
    tags = set()
    caps = capitalised_inner_words(text)
    if any(w in PLACES for w in caps) or any(p in text for p in PLACES):
        tags.add("places")
    if any(w not in PLACES and w not in NOT_NAMES for w in caps):
        tags.add("names")
    if re.search(NUMBER_WORDS, text, re.I) or re.search(HINDI_NUMBERS, text):
        tags.add("numbers")
    if re.search(CURRENCY, text, re.I):
        tags.add("currency")
    if re.search(ACRONYM, text):
        tags.add("acronyms")
    if re.search(DATES_TIMES, text, re.I):
        tags.add("dates_times")
    group = language_group(row)
    if group != "en":
        tags.add(group)
    return [t for t in TAG_ORDER if t in tags]


def estimated_seconds(text: str) -> float:
    return len(text) / CHARS_PER_SECOND


def too_long(text: str) -> list[str]:
    reasons = []
    if estimated_seconds(text) > MAX_SECONDS * 0.95:  # a little margin: estimates are rough
        reasons.append(f"~{estimated_seconds(text):.1f}s spoken")
    if len(text) > MAX_TEXT_CHARS:
        reasons.append(f"{len(text)} chars > {MAX_TEXT_CHARS}")
    return reasons


def interleave(groups: dict[str, list]) -> list:
    """Merge lists so each group is spread evenly through the output, in proportion to its size."""
    keyed = []
    for items in groups.values():
        n = len(items)
        for i, item in enumerate(items):
            keyed.append(((i + 0.5) / n, item))
    keyed.sort(key=lambda pair: pair[0])
    return [item for _, item in keyed]


def build_order(rows: list[dict]) -> list[dict]:
    groups: dict[str, list] = {"en": [], "hinglish": [], "hi": []}
    for row in rows:
        tags = tags_for(row)
        groups[language_group(row)].append({**row, "tags": tags})
    for items in groups.values():
        items.sort(key=lambda r: -len(r["tags"]))  # stable: keeps original order on ties
    return interleave({k: v for k, v in groups.items() if v})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", default=str(ROOT / "recording" / "prompts.tsv"))
    ap.add_argument("--out", default=str(ROOT / "recording" / "order.tsv"))
    args = ap.parse_args()

    with open(args.prompts, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    ordered = build_order(rows)

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["id", "lang", "text", "group", "tags", "est_seconds"])
        for r in ordered:
            w.writerow([r["id"], r["lang"], r["text"], language_group(r), ",".join(r["tags"]),
                        f"{estimated_seconds(r['text']):.1f}"])

    flagged = [(r["id"], too_long(r["text"])) for r in ordered if too_long(r["text"])]
    first = ordered[:220]
    mix = {g: sum(language_group(r) == g for r in first) for g in ("en", "hinglish", "hi")}
    print(f"wrote {len(ordered)} prompts to {args.out}")
    print(f"first 220 (about session 1): {mix}")
    print(f"flagged as maybe too long: {len(flagged)}")
    for pid, reasons in flagged:
        print(f"  {pid}: {', '.join(reasons)}")


if __name__ == "__main__":
    main()
