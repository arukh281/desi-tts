# Benchmark

`benchmark.tsv` is the fixed test set. Every model is scored on exactly these sentences, and none of them are used for training (`scripts/check_overlap.py` checks this).

- `id`: row id
- `lang`: `en`, `hinglish` or `hi`
- `category`: the main thing being tested, one of conversational, names, places, numbers, dates_times, currency, acronyms, questions, long
- `tags`: other categories the sentence also covers, comma-separated, can be empty
- `text`: written the way a real system would send it, e.g. ₹12,750, 4:30 PM, KYC
- `check`: the hard parts a listener should verify, e.g. `₹12,750; EMI`
