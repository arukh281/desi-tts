# Benchmark

`benchmark.tsv` is the fixed test set. I score every model on exactly these 88 sentences, and none of them are used for training (`scripts/check_overlap.py` checks this). The sentences were drafted with AI, then I reviewed them and froze the file before any training.

Columns:
- `id`: row id
- `lang`: `en`, `hinglish` or `hi`
- `category`: the main thing being tested, one of conversational, names, places, numbers, dates_times, currency, acronyms, questions, long
- `tags`: other things the sentence also covers, comma-separated, can be empty
- `text`: written the way a real system would send it, for example ₹12,750, 4:30 PM, KYC
- `check`: the hard parts a listener should check, for example `₹12,750; EMI`

## Other text sets

- `hinglish_deva.tsv`: Devanagari versions of the 16 Hinglish rows, used only for scoring. They were AI-drafted and I checked them.
- `hinglish_heldout.tsv`, `long_heldout.tsv`, `names_heldout.tsv`: held-out sets I used for choices, never for training. Like the benchmark, they were AI-drafted and I reviewed them. `names_heldout.tsv` has 20 English sentences with Indian names and places that appear in neither the benchmark nor my prompts.
