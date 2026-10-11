# desi-tts

I fine-tuned XTTS-v2 on 24 minutes of my own voice so it speaks Indian English, Hindi and Hinglish.

I recorded the data and made the final calls. The code and most of the writing were done with a lot of AI help (Claude).

- Report (5 pages): [report/REPORT.md](report/REPORT.md)
- Experiment table: [report/EXPERIMENTS.md](report/EXPERIMENTS.md)
- Samples, baseline against final, works on a phone: https://arukh281.github.io/desi-tts/ (source in [docs/](docs))
- Daily log: [DAYS.md](DAYS.md)

## Result

The final model reads Hinglish far better and sounds like me. English and Hindi errors stay about the same, and Indian names are still its weak spot.

The table is the 88-sentence benchmark, with Whisper large-v3's character error rate (CER, lower is better) averaged over 3 seeds. Hinglish is a rough score: Hindi Whisper against AI-drafted Devanagari references. Sim is speaker similarity to my reference clip.

| | pretrained XTTS-v2 + normaliser | final | difference, 95% interval |
|---|---|---|---|
| CER, all | 0.125 | 0.061 | −0.064 [−0.110, −0.025], real |
| CER, Indian English | 0.037 | 0.043 | +0.006 [−0.014, +0.029], within noise |
| CER, Hindi | 0.198 | 0.102 | −0.096 [−0.331, +0.054], within noise |
| CER, Hinglish | 0.400 | 0.098 | −0.303 [−0.392, −0.215], real |
| sim, English | 0.502 | 0.730 | my real voice: 0.763 |

The final model is XTTS-v2 with its GPT part fine-tuned on my recordings (step 1500). In front of it, a normaliser writes numbers, ₹, dates and acronyms as words, and Hinglish is sent in Devanagari using IndicXlit. On a T4 it is a 1.8 GB model using 2.5 GB of GPU memory, with a real-time factor of 0.51. Streaming, the first audio comes after 0.43 s, or about 0.57 s for Hinglish.

Files: [results/summary.md](results/summary.md), [results/bootstrap_E1_vs_Final.txt](results/bootstrap_E1_vs_Final.txt), [latency](results/runs/extras/latency/profile.json)

## Repo layout

```
benchmark/       frozen 88-sentence benchmark, Devanagari references for Hinglish, held-out sets
configs/         sampling settings (same for every model), name respelling list
data_prep/       clip checks, cleaning, Whisper transcript check, split, XTTS metadata, dataset card
docs/            sample page (index.html) and its 27 small MP3s
eval/            Whisper CER, speaker similarity, duration checks, 3-seed runs, bootstrap, summaries, error analysis
infer/           synthesis, latency profile, transliteration timing
kaggle/          Kaggle kernel template (runs one repo script at an exact commit)
recording/       recording tool for the Mac, recording order, reference sentences
report/          REPORT.md and EXPERIMENTS.md
results/         the small CSV, JSON and text files behind every reported number (no audio)
scripts/         Kaggle runner, dataset upload, env check, overlap check, sample page builder, table rebuild
text/            normaliser (written to spoken form) and Hinglish to Devanagari
train/           GPT fine-tuning, checkpoint selection on val only, export, one-run experiment
tests/           unit tests
DAYS.md          short daily log
```

## How to reproduce

You can rebuild every number table from this repo alone. My voice recordings and the trained weights are private, so training and synthesis can't be rerun without them. The dataset and weights are available to reviewers on request.

To rebuild the tables, run `scripts/reproduce_tables.sh`. It needs no GPU and no private data, and afterwards `git status results/` should show no changes. The speaker-consistency, prosody, question-pitch and names-on-real-takes tables need the private audio; their scripts are in `eval/` and their outputs are in `results/`.

The full pipeline runs GPU steps on a free Kaggle T4 through `scripts/kaggle_run.sh`. It runs a repo script in a private kernel at the current pushed commit and downloads the output to `outputs/kaggle/<run>/`. My recordings and processed data are private Kaggle datasets (`desi-tts-own-voice`, `desi-tts-reference`), so the data steps need your own recordings. Steps 8 and 9 run on the files in `results/` with no GPU.

1. Setup (Mac): `python3 -m venv .venv && .venv/bin/pip install -r recording/requirements.txt -r data_prep/requirements.txt`
2. Tests: `.venv/bin/pip install pytest && .venv/bin/python -m pytest -q tests`. 46 pass, and one normaliser check is skipped unless the private `recording/prompts.tsv` is there. No GPU needed; the data-prep test downloads the 45 MB IndicXlit model once.
3. Record (Mac): `python recording/order.py`, then `python recording/record.py --reference` and `python recording/record.py --session 1` (and `--session 2`)
4. Prepare data (Mac): `python data_prep/prepare.py`, upload with `scripts/upload_dataset.sh data/processed desi-tts-own-voice`, run `data_prep/asr_check.py` on Kaggle, copy `asr.csv` into `data/processed/`, then `python data_prep/finalize.py && python data_prep/dataset_card.py` and upload again
5. Fine-tune one experiment (Kaggle), which trains, picks a checkpoint on val and runs the benchmark:
   `DATASETS="desi-tts-own-voice desi-tts-reference" scripts/kaggle_run.sh e3 train/run_experiment.py --name e3 --data /tmp/in/desi-tts-own-voice --ref <3 ref clips> --keep-model -- --lr 5e-6 --batch 2 --grad-accum 8 --max-steps 2000 --eval-step 200 --seed 1234`
6. Extras (Kaggle): `eval/extras.py --model <export dir> ...` for the Hinglish 2x2, held-out Hinglish, real-voice reference and latency profile
7. Final comparison (Kaggle): `KERNELS="<e3 run> <e2b run>" scripts/kaggle_run.sh final eval/final_seeds.py --ref <3 ref clips>` (E1, E2b, E3 and the Hinglish sets, 3 seeds)
8. Numbers (no GPU): copy the CSVs into `results/final/`, then `python eval/build_final.py`, `python eval/summary.py` and `python eval/bootstrap.py --a E1=results/final/E1 --b Final=results/final/Final`
9. Error analysis (no GPU): `python eval/error_analysis.py --synth-dir <run> --benchmark benchmark/benchmark.tsv` and `python eval/names_seen.py E3=results/final/E3/seed1,...`
10. Synthesise with the final pipeline: `python infer/synthesize.py --model <export dir> --ref <ref clips> --normalize on --hinglish-route deva`

Datasets and model weights aren't in this repo. XTTS-v2's weights are non-commercial and Coqui has shut down, so a commercial deployment needs a different base model (report, section 8).
