# Benchmark

`benchmark.tsv` is the fixed test set. Every model is scored on exactly these sentences, and none of them are used for training (`scripts/check_overlap.py` checks this).

- `id`: row id
- `lang`: `en`, `hinglish` or `hi`
- `category`: the main thing being tested, one of conversational, names, places, numbers, dates_times, currency, acronyms, questions, long
- `tags`: other categories the sentence also covers, comma-separated, can be empty
- `text`: written the way a real system would send it, e.g. ₹12,750, 4:30 PM, KYC
- `check`: the hard parts a listener should verify, e.g. `₹12,750; EMI`

## Other text sets

- `hinglish_deva.tsv`: AI-drafted Devanagari references for scoring the 16 Hinglish rows (checked by me).
- `hinglish_heldout.tsv`, `long_heldout.tsv`, `names_heldout.tsv`: held-out sets used only for choices, never for training. Like the benchmark, they were AI-drafted and reviewed by me. `names_heldout.tsv` has 20 English sentences with Indian names and places that appear in neither the benchmark nor the prompts.
