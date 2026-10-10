# desi-tts

Fine-tuning XTTS-v2 on 24 minutes of my own voice for Indian English, Hindi and Hinglish.

Code and writing drafted with AI assistance; recordings, decisions and listening tests are mine.

- **Report:** [report/REPORT.md](report/REPORT.md) (5 pages)
- **Experiment table:** [report/EXPERIMENTS.md](report/EXPERIMENTS.md)
- **Samples (baseline vs final, phone-friendly):** https://arukh281.github.io/desi-tts/ (source in [docs/](docs))
- **Daily log:** [DAYS.md](DAYS.md)

## Result in one table

Benchmark of 88 sentences, Whisper large-v3 character error rate (lower is better), averaged over 3 seeds. Hinglish is a rough score (Hindi Whisper against hand-written Devanagari). Sim = speaker similarity to my reference clip.

| | pretrained XTTS-v2 + normaliser | **final** | difference, 95% interval |
|---|---|---|---|
| CER, all | 0.125 | **0.061** | −0.064 [−0.110, −0.025] |
| CER, Indian English | 0.037 | 0.043 | +0.006 [−0.014, +0.029] (no change) |
| CER, Hindi | 0.198 | 0.102 | −0.096 [−0.331, +0.054] (not significant) |
| CER, Hinglish | 0.400 | **0.098** | −0.303 [−0.392, −0.215] |
| sim, English | 0.502 | 0.730 | my real voice: 0.763 |

Final = XTTS-v2 with its GPT part fine-tuned on my recordings (step 1500), a text normaliser for numbers, ₹, dates and acronyms, and Hinglish sent to the model in Devanagari via IndicXlit. On a T4: 1.8 GB model, 2.5 GB VRAM, real-time factor 0.51, first audio after 0.43 s streaming (about 0.57 s for Hinglish). Main remaining weakness: Indian names. Sources: [results/summary.md](results/summary.md), [results/bootstrap_E1_vs_Final.txt](results/bootstrap_E1_vs_Final.txt), [latency](results/runs/extras/latency/profile.json).

## Repo layout

```
benchmark/       frozen 88-sentence benchmark, Devanagari Hinglish for scoring, held-out Hinglish and long sets
configs/         sampling settings, the same for every model compared
data_prep/       clip checks, cleaning, Whisper transcript check, split, XTTS metadata, dataset card
docs/            sample page (index.html) and its 27 small MP3s
eval/            Whisper CER, speaker similarity, duration checks; 3-seed runs, bootstrap, summaries, error analysis
infer/           synthesis, latency profile, transliteration timing
kaggle/          Kaggle kernel template (runs one repo script at an exact commit)
recording/       recording tool for the Mac, recording order, reference sentences
report/          REPORT.md and EXPERIMENTS.md
results/         the small CSV/JSON/text files behind every reported number (no audio)
scripts/         Kaggle runner, private dataset upload, env check, overlap check, sample page builder
text/            normaliser (written -> spoken form) and Hinglish -> Devanagari
train/           GPT fine-tuning, checkpoint selection on val only, export, one-run experiment
tests/           unit tests
DAYS.md          short daily log
```

## How to reproduce

GPU steps run on a free Kaggle T4 through `scripts/kaggle_run.sh`, which runs a repo script in a private kernel at the current pushed commit and downloads the output to `outputs/kaggle/<run>/`. My recordings and the processed data are private Kaggle datasets (`desi-tts-own-voice`, `desi-tts-reference`), so the data steps need your own recordings. Everything from step 8 on runs on the result files in `results/` without a GPU.

1. **Setup (Mac):** `python3 -m venv .venv && .venv/bin/pip install -r recording/requirements.txt -r data_prep/requirements.txt`
2. **Tests:** `.venv/bin/python -m unittest discover -s tests`
3. **Record (Mac):** `python recording/order.py`, then `python recording/record.py --reference` and `python recording/record.py --session 1` (and `--session 2`)
4. **Prepare data (Mac):** `python data_prep/prepare.py`, upload with `scripts/upload_dataset.sh data/processed desi-tts-own-voice`, run `data_prep/asr_check.py` on Kaggle, copy `asr.csv` into `data/processed/`, then `python data_prep/finalize.py && python data_prep/dataset_card.py` and upload again
5. **Fine-tune one experiment (Kaggle):** train, pick a checkpoint on val, run the benchmark:
   `DATASETS="desi-tts-own-voice desi-tts-reference" scripts/kaggle_run.sh e3 train/run_experiment.py --name e3 --data /tmp/in/desi-tts-own-voice --ref <3 ref clips> --keep-model -- --lr 5e-6 --batch 2 --grad-accum 8 --max-steps 2000 --eval-step 200 --seed 1234`
6. **Extras (Kaggle):** `eval/extras.py --model <export dir> ...` for the Hinglish 2x2, held-out Hinglish, real-voice ceiling and latency profile
7. **Final comparison (Kaggle):** `KERNELS="<e3 run> <e2b run>" scripts/kaggle_run.sh final eval/final_seeds.py --ref <3 ref clips>` (E1, E2b, E3 and the Hinglish sets, 3 seeds)
8. **Numbers (no GPU):** copy the CSVs into `results/final/`, then
   `python eval/build_final.py`, `python eval/summary.py`, and `python eval/bootstrap.py --a E1=results/final/E1 --b Final=results/final/Final`
9. **Error analysis (no GPU):** `python eval/error_analysis.py --synth-dir <run> --benchmark benchmark/benchmark.tsv` and `python eval/names_seen.py E3=results/final/E3/seed1,...`
10. **Synthesise with a trained model:** `python infer/synthesize.py --model <export dir> --ref <ref clips> --normalize on --hinglish-route deva` (the final pipeline)

Datasets and model weights aren't in this repo. XTTS-v2's licence is non-commercial.
