"""Score pasted results from the listening test page against the private key.

Paste each listener's WhatsApp text into one file (blocks start with
"desi-tts listening v1 | <name>"). Results are reported twice: for the owner
(--owner, default Aradhya, who knows his own voice and the project) and for
everyone else, because an owner's ratings aren't blind in the same way.

For each pair type it counts which system was preferred for "more natural" and
"more like Aradhya" (ties and "don't know" counted separately), and how often
listeners marked a wrong word in each system. Repeated pairs (A/B swapped) give
a consistency score: did the listener pick the same system both times?

Usage:  python eval/listening_score.py --key KEY.json --results pasted.txt [--owner Aradhya]
"""
import argparse
import json
import re
from collections import Counter, defaultdict

LINE = re.compile(r"^(P\d\d) nat=(\S) like=(\S) wa=(\S) wb=(\S)")


def parse(text: str) -> dict[str, dict[str, dict[str, str]]]:
    listeners, current = {}, None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("desi-tts listening"):
            current = line.split("|", 1)[1].strip() if "|" in line else f"listener{len(listeners) + 1}"
            listeners[current] = {}
        elif current and (m := LINE.match(line)):
            pid, nat, like, wa, wb = m.groups()
            listeners[current][pid] = {"nat": nat, "like": like, "wa": wa, "wb": wb}
    return listeners


def system(choice: str, pair: dict) -> str:
    return {"A": pair["A"], "B": pair["B"], "S": "same", "X": "don't know", "-": "no answer"}[choice]


def report(title: str, listeners: dict, key: dict) -> None:
    print(f"\n## {title}: {', '.join(listeners) or 'nobody yet'}")
    if not listeners:
        return
    by_kind = defaultdict(lambda: {"nat": Counter(), "like": Counter(), "wrong": Counter(), "heard": Counter()})
    agree, compared = Counter(), Counter()
    for answers in listeners.values():
        for pid, a in answers.items():
            pair = key[pid]
            if pair["repeat_of"]:
                original = next(p for p, k in key.items() if k["row"] == pair["row"] and not k["repeat_of"])
                if original in answers:
                    for q in ("nat", "like"):
                        first, again = system(answers[original][q], key[original]), system(a[q], pair)
                        if first not in ("no answer",) and again not in ("no answer",):
                            compared[q] += 1
                            agree[q] += first == again
                continue
            stats = by_kind[pair["kind"]]
            stats["nat"][system(a["nat"], pair)] += 1
            stats["like"][system(a["like"], pair)] += 1
            for side in ("A", "B"):
                ans = a["wa" if side == "A" else "wb"]
                if ans in "YN":
                    stats["heard"][pair[side]] += 1
                    stats["wrong"][pair[side]] += ans == "Y"
    for kind, s in by_kind.items():
        print(f"\n{kind}")
        print(f"  more natural:     {dict(s['nat'])}")
        print(f"  more like Aradhya: {dict(s['like'])}")
        for name in sorted(s["heard"]):
            print(f"  word wrong in {name}: {s['wrong'][name]}/{s['heard'][name]} judgements")
    for q, label in (("nat", "natural"), ("like", "like Aradhya")):
        if compared[q]:
            print(f"  consistency ({label}) on repeated pairs: {agree[q]}/{compared[q]} same system picked")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--key", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--owner", default="Aradhya")
    args = ap.parse_args()
    key = json.load(open(args.key))["pairs"]
    listeners = parse(open(args.results, encoding="utf-8").read())
    owner = {n: a for n, a in listeners.items() if n.lower() == args.owner.lower()}
    others = {n: a for n, a in listeners.items() if n.lower() != args.owner.lower()}
    report("Owner (not blind to his own voice)", owner, key)
    report("Other listeners", others, key)


if __name__ == "__main__":
    main()
