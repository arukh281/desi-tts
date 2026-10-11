#!/usr/bin/env bash
# Rebuild every table that comes from results/ alone (no audio, no private data, no GPU).
# Afterwards `git status results/` should show no changes: that is the check.
#
#   scripts/reproduce_tables.sh
#
# Tables that need my private audio or recordings (consistency, prosody, question
# pitch, names on my real takes, listening scores, transliteration timing) are
# made by their own scripts; see the README.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
F=results/final
R=results/runs

"$PY" eval/build_final.py >/dev/null
"$PY" eval/summary.py >/dev/null

boot() { "$PY" eval/bootstrap.py --a "$1=$F/$1" --b "$2=$F/$2" > "results/bootstrap_${1}_vs_${2}.txt"; }
boot E1 E3; boot E1 Final; boot E3 Final; boot E1_deva E3_deva; boot E3 E3_deva
boot E1_heldout_deva E3_heldout_deva; boot E3_heldout_en E3_heldout_deva; boot E1_heldout_en E3_heldout_deva
boot E2b E3
"$PY" eval/bootstrap.py --a "E3_deva=$F/E3_heldout_deva" --b "E4=$R/E4/heldout" > results/bootstrap_E3_heldout_deva_vs_E4.txt

G=(--group lang=en,category=names --group lang=en,category=places --group category=names --group lang=hinglish,category=names --group lang=hi,category=names)
for pair in "E1 E3" "E1 Final" "E3 Final"; do
  set -- $pair
  "$PY" eval/bootstrap.py --a "$1=$F/$1" --b "$2=$F/$2" "${G[@]}" > "results/bootstrap_names_${1}_vs_${2}.txt"
done

{ echo "<!-- python eval/names_seen.py, 3 seeds pooled. Final = E3 on these rows except the Hinglish ones (b067 b068 b071 b072), which go through the Devanagari route -->"
  "$PY" eval/names_seen.py "E1=$F/E1/seed1,$F/E1/seed2,$F/E1/seed3" "E3=$F/E3/seed1,$F/E3/seed2,$F/E3/seed3"
} > results/names_seen.md

{ echo "<!-- python eval/compare.py, single seed (1234), one T4 run each; crashed rows count as CER 1.0 -->"
  "$PY" eval/compare.py --run E0=$R/E0 --run E1=$R/E1_single --run E2raw=$R/E2/bench_norm_off --run E2=$R/E2/bench_norm_on \
    --run E2b_r1=$R/E2b_run1/bench_norm_on --run E3_r1=$R/E3_run1/bench_norm_on --run E3_r1_deva=$R/E3_run1/bench_deva
} > results/single_seed.md 2>&1

git status --short results/ | sed 's/^/changed: /'
echo "done"
