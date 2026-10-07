# desi-tts

Fine-tuning an open TTS model for Indian English, with Hindi / Hinglish support.

Code and writing drafted with AI assistance; recordings, decisions and listening tests are mine.

This is still a work in progress, so things will move around a bit. I'll keep updating this README as I go.

## What I'm trying to do

Take an existing pretrained TTS model and make it better at the things Indian voice apps actually need:

- Indian English accent and natural prosody
- Indian names and places
- numbers, dates, currency (₹, lakh/crore) and acronyms like KYC, EMI, UPI
- Hinglish, e.g. "Aapka payment successfully receive ho gaya hai"

## Plan

- [ ] Collect and clean ~2-3 hours of speech data
- [ ] Pick a base model and generate baseline audio on a fixed benchmark set
- [ ] Fine-tune on the cleaned data
- [ ] Run a few experiments (data filtering, checkpoint selection, targeted data)
- [ ] Compare baseline vs fine-tuned: listening tests, pronunciation, intelligibility, speaker consistency
- [ ] Find one clear failure and fix it
- [ ] Measure inference speed and write up the report

## Repo layout

```
benchmark/                  fixed test set (88 sentences) and what each column means
configs/sampling.json       XTTS sampling settings, the same for every model compared
data_prep/                  clip checks and cleaning, Whisper transcript check, split, XTTS metadata, dataset card
eval/                       Whisper CER/WER, speaker similarity (ECAPA), duration checks, evaluate.py runs all three
infer/synthesize.py         synthesises a text set and times it (RTF, time to first audio)
infer/fixture.tsv           6 fixed sentences for quick checks
kaggle/kernel_template.py   Kaggle kernel that runs one repo script at an exact commit
recording/                  recording tool (Mac), recording order, reference sentences
scripts/                    Kaggle runner, private dataset upload, env check, benchmark overlap check
text/normalize.py           written form -> spoken form (rupees, numbers, dates, times, acronyms)
train/                      GPT fine-tuning, checkpoint export, training smoke test
tests/                      unit tests (recording, overlap check, normaliser, data prep)
requirements.txt            Kaggle-side pins (coqui-tts, transformers, speechbrain)
DAYS.md                     short daily log of what I did and why
```

## How to reproduce

GPU steps run on a free Kaggle T4 through `scripts/kaggle_run.sh`, which pushes a private kernel at the current commit (push it first) and downloads the output to `outputs/kaggle/<run>/`. My recordings and the processed data are private Kaggle datasets, never public.

1. **Setup (Mac):**
   `python3 -m venv .venv && .venv/bin/pip install -r recording/requirements.txt -r data_prep/requirements.txt`
2. **Tests:** `.venv/bin/python -m unittest discover -s tests`
3. **Kaggle check:** `scripts/kaggle_run.sh env-check scripts/env_check.py`
4. **Record (Mac):** `python recording/order.py`, then `python recording/record.py --reference` and `python recording/record.py --session 1`
5. **Prepare data (Mac):** `python data_prep/prepare.py`
6. **Transcript check (Kaggle):** upload with `scripts/upload_dataset.sh data/processed desi-tts-own-voice`, run `data_prep/asr_check.py` on it, copy `asr.csv` into `data/processed/`
7. **Finalise (Mac):** `python data_prep/finalize.py && python data_prep/dataset_card.py`, then upload `data/processed` again
8. **Baseline (Kaggle):** `eval/evaluate.py --name baseline --model pretrained --ref <reference clips> --normalize off`, and again with `--normalize on`
9. **Fine-tune (Kaggle):** `train/train_gpt.py --data <dataset> ...`, then `train/export.py`
10. **Evaluate the fine-tuned model (Kaggle):** `eval/evaluate.py --name finetuned --model <export dir> --ref <same clips> --normalize on`

Pipeline smoke tests on stock-voice data: `train/smoke_test.py` (training, resume, export) and `eval/evaluate.py --speaker ...` (evals).

## Notes

Datasets and model weights aren't stored in this repo.
