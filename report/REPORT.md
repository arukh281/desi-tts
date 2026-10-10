# desi-tts: fine-tuning XTTS-v2 on my own voice for Indian English, Hindi and Hinglish

Aradhya Khandelwal, October 2026. Code and writing drafted with AI assistance; recordings, decisions and listening tests are mine.

Every number below comes from a file in [`results/`](../results); the file is named next to each table. The experiment table is in [EXPERIMENTS.md](EXPERIMENTS.md) and audio samples are on the [sample page](https://arukh281.github.io/desi-tts/).

## 1. Summary

I fine-tuned the GPT part of XTTS-v2 on 24 minutes of my own speech and added two text front-ends: a normaliser for numbers, currency, dates and acronyms, and a route that sends Hinglish to the model in Devanagari. On an 88-sentence benchmark (3 seeds), character error rate (CER) falls from 0.125 for the pretrained model with the same normaliser to 0.061 (−0.064, 95% interval [−0.110, −0.025]), almost all from Hinglish (0.400 → 0.098); English and Hindi don't change beyond noise. English speaker similarity rises from 0.50 to 0.73 (my real recordings: 0.76), and listeners picked the fine-tuned model as more like me in 15 of 16 pairs. Names are the main weakness: English name rows got worse (0.030 → 0.106, +0.076 [+0.005, +0.193]).

## 2. Data

**Why my own voice.** None of the open sets I checked (IndicTTS, SYSPIN, Rasa, LIMMITS, Svarah) has one clean speaker across Indian English, Hinglish and Hindi, and mixing speakers would blur the voice. I also know the consent.

**Recording.** MacBook Air built-in mic, 48 kHz, two sessions (8 and 10 Oct; 16.4 and 7.8 minutes used), with a small recorder that shows a level meter and warns on clipping, low level and long takes. I planned about an hour and three sessions; I stopped after two, so the set is **24 minutes** ([dataset_card](../results/dataset_card.md)).

| group | sent to XTTS as | clips | minutes | train / val |
|---|---|---|---|---|
| Indian English | en | 154 | 12 | 146 / 8 |
| Hinglish (Roman script) | en | 82 | 5.4 | 78 / 4 |
| Hindi (Devanagari) | hi | 97 | 6.6 | 92 / 5 |
| total | | 333 | 24 | 316 / 17 |

**Cleaning.** Each prompt is its own take, so no segmentation or alignment was needed. Takes are checked for clipping, SNR (median 37 dB), long pauses and length; Enter-key clicks and edge silence are cut; loudness is normalised to −23 LUFS and audio resampled to 22.05 kHz. The transcript is the prompt as written: already spoken form (no digits or symbols), punctuation and case kept for phrasing; XTTS's tokenizer does its standard cleaning. Whisper large-v3 then checks every take. Two lessons: Whisper forced to English *translates* Hinglish, so that CER is report-only; and every automatic drop but one was Whisper mishearing a correct take, so drops need a listen. 333 of 334 takes are used (p0293 was cut off); 45 are flagged but kept (35 borderline transcripts, 6 low SNR, 5 long pauses; one has two).

**Provenance.** The 650 prompts were written with AI help: p0001-p0480 by a template script, p0481-p0650 drafted by an AI assistant as a Hindi/Hinglish top-up, and 24 prompts changed to masculine first person on my instruction. I read Hindi from an AI-written romanisation shown by the recorder, while the saved transcript stays in Devanagari. The 88 benchmark sentences and 3 reference sentences are kept out of training, which a leakage check confirms. The audio and the prompt list stay private.

## 3. Model and text

**Model.** XTTS-v2 (coqui-tts 0.27.5). It already speaks English and Hindi, clones a voice from short reference audio and can stream. I fine-tune only its GPT part: 441 M of 518 M parameters ([training_summary](../results/runs/E3_run2/train/train_summary.json)). The GPT decides which sounds to make, so it carries accent and pronunciation. The audio codec and the HiFi-GAN decoder stay frozen, which keeps training within a 16 GB T4. XTTS-v2's licence is non-commercial, which matters for production (section 8).

**Text representation.** Text goes in as graphemes: XTTS's BPE tokenizer splits characters into subword tokens, with no phonemes. The GPT predicts discrete acoustic tokens from XTTS's codec (a discrete VAE), and the HiFi-GAN decoder turns the GPT's output into audio. That turned out to help: wrong readings can often be fixed in the text before the model sees it.
- **Normaliser** (`text/normalize.py`): ₹, numbers, lakh/crore, dates, times and acronyms become the words a person would say ("₹12,750" → "twelve thousand seven hundred and fifty rupees", "KYC" → "K Y C"). It reproduces all 64 expected spoken forms in the benchmark. Without it, pretrained XTTS crashes on 8 of 12 Hindi rows, because its own digit expander fails on Hindi.
- **Hinglish route** (`text/translit.py`): Roman Hinglish is turned into Devanagari with IndicXlit (AI4Bharat, MIT licence) plus a lexicon of 178 number and everyday words, then sent to XTTS as Hindi. Section 7 covers why.

## 4. Training

Final run E3: sessions 1+2 (316 train / 17 val clips), learning rate 5e-6 (the coqui recipe default, low so the model keeps its English and Hindi), batch 2 with gradient accumulation 8 (16 clips per update), a 2000-step target (2054 steps run, 13 epochs), fp32. A step here is one micro-batch of 2 clips, so 2054 steps are about 257 optimizer updates, seed 1234. Weights are saved at 500, 1000, 1500 and 2000 steps. The checkpoint is chosen on the 17 val clips only: lowest val loss, among checkpoints whose val CER is within 0.05 of the best and that have no more than 2 extra duration flags. That picked **step 1500**. Step 2000 had the lowest loss but a val CER of 0.135, a sign of babbling ([selection](../results/runs/E3_run2/select/selection.csv)).

**Time.** 2054 steps (median 0.44 s per step), **16.4 minutes** of summed step time on one T4, peak 9.4 GB VRAM ([training](../results/training.md)). The whole Kaggle job took about 50 minutes ([run_info](../results/runs/E3_run2/run_info.json)). I trained E3 twice with the same seed; run 2 gave exactly the same selection table as run 1, so the pick is reproducible ([run 1](../results/runs/E3_run1/select/selection.csv), [run 2](../results/runs/E3_run2/select/selection.csv)).

## 5. Benchmark and experiments

**Benchmark.** 88 sentences, drafted with AI, reviewed by me and frozen (commit dc343ba, 7 Oct) before any training: 60 Indian English in 9 categories (conversational, names, places, numbers, dates and times, currency, acronyms, questions, long), 16 Hinglish and 12 Hindi. Names and places are tagged *seen* (in my prompts) or *unseen*. The benchmark is used only for final comparisons. Every choice (checkpoint, sampling, splitting, E4) was made on val clips or on two held-out sets: 10 extra Hinglish sentences and 5 long sentences.

**Metrics.**
- *Intelligibility:* Whisper large-v3 CER against the spoken form. Hinglish is scored with Hindi Whisper against AI-drafted Devanagari references, because English Whisper translates it. That makes Hinglish CER a rough signal.
- *Voice:* ECAPA speaker similarity to my reference clip in the same language. Similarity drops across languages even for one stock voice, so I only compare within a language.
- *Delivery:* duration checks for stretched, rushed or babbling clips.
- *Noise:* final numbers average 3 seeds. A paired bootstrap over sentences (10,000 resamples) gives a 95% interval, and I call a difference real only if the interval excludes 0. A crashed sentence counts as CER 1.0, so a system can't look better by skipping what it can't say.

**Experiments** (details in [EXPERIMENTS.md](EXPERIMENTS.md)):
- **E0** is the pretrained model on raw text; it crashes on Hindi digits. **E1** is the pretrained model with the normaliser, and is the baseline.
- **E2** fine-tuned on session 1 and picked the checkpoint by val CER. It **failed**: it chose step 250, which sounded robotic and not like me, even though its English speaker similarity was already high (0.718). So ECAPA similarity tracks who is speaking, not how natural it sounds. CER on 8 val clips was too noisy to choose by, so I switched to the val-loss rule above.
- **E2b** repeats E3's recipe on session 1 only (16 vs 24 min), trained twice. Both runs picked step 1000; run 2 wasn't bit-identical (val loss at that step within 0.002). Against E3 with 3 seeds, the extra 8 minutes measurably helped only Hinglish (E3 − E2b = −0.045 [−0.076, −0.017]); English and Hindi are within noise ([bootstrap_E2b_vs_E3](../results/bootstrap_E2b_vs_E3.txt); run 1 in [EXPERIMENTS.md](EXPERIMENTS.md)).
- **E4** trains Hinglish as Devanagari, **E5** sweeps sampling settings and **E6** splits long sentences. None of them beat noise, so none were adopted.

## 6. Results

3 seeds, normaliser on, my 3 reference clips ([summary](../results/summary.md), bootstraps in [results/](../results)).

| | E1 pretrained | E3 fine-tuned | **Final** (E3 + Hinglish route) | my real voice (val clips, reference point) |
|---|---|---|---|---|
| CER, all 88 | 0.125 | 0.084 | **0.061** | – |
| CER, English (60) | 0.037 | 0.043 | 0.043 | 0.100 (8 clips) |
| CER, Hindi (12) | 0.198 | 0.102 | 0.102 | 0.061 (5 clips) |
| CER, Hinglish (16, rough) | 0.400 | 0.222 | **0.098** | not comparable* |
| speaker sim en / hi / Hinglish | 0.502 / 0.643 / 0.589 | 0.730 / 0.706 / 0.720 | 0.730 / 0.706 / 0.738 | 0.763 / 0.615 / 0.729 |

\*Real Hinglish was scored with English Whisper (0.521), which translates it. The Final row's speaker similarity is E3's for English and Hindi and the Devanagari route's for Hinglish.

| Final − E1 | difference | 95% interval | real? |
|---|---|---|---|
| all | −0.064 | [−0.110, −0.025] | yes |
| English | +0.006 | [−0.014, +0.029] | no |
| Hindi | −0.096 | [−0.331, +0.054] | no (12 rows, wide) |
| Hinglish | −0.303 | [−0.392, −0.215] | yes |

Fine-tuning alone (E3 − E1) is −0.041 [−0.083, −0.009] overall and −0.178 [−0.247, −0.118] on Hinglish. By category, E1 → Final: questions 0.291 → 0.026, currency 0.103 → 0.020, dates and times 0.189 → 0.093, numbers 0.169 → 0.105, places 0.073 → 0.091, names 0.107 → 0.095 ([bootstrap_E1_vs_Final](../results/bootstrap_E1_vs_Final.txt)). The names gain comes only from the 3 Hinglish name rows the route fixes (0.355 → 0.084); English name rows get worse ([bootstrap_names](../results/bootstrap_names_E1_vs_Final.txt)).

The real-voice row is my own val recordings with the same Whisper and ECAPA settings: a reference point, not a ceiling (different sentences, 17 clips). English synthetic similarity (0.730) is close to it (0.763); Hindi synthetic is above it (0.706 vs 0.615), probably because synthesis is conditioned on the reference clip.

**Listening test.** One joint session: three people listened together and agreed each answer, so it is one consensus judgement, not three independent ones. The page was blind and in random order: 22 A/B pairs, 16 E1 vs E3, 4 Hinglish Roman vs Devanagari on E3, and 2 repeats with A and B swapped. The E3 audio was from E3 run 1, whose training matches run 2 exactly ([listening](../results/listening_joint_session1.md)).

Results: more like Aradhya, E3 15 of 16 (1 same, E1 0); more natural, E3 11, E1 4, same 1; a word wrong or unclear, E1 11/16 and E3 9/16; Hinglish Devanagari vs Roman, Devanagari 4/4 on both questions, with a wrong word in Roman 4/4 and Devanagari 1/4; repeated pairs answered the same way 2/2.

The voice change is clear. Both systems still get words wrong often, which matches the names result below.

**Speaker consistency and prosody** (seed-1234 audio; my real takes are different sentences, so they are a loose reference; [consistency](../results/consistency.md), [prosody](../results/prosody.md)):

| | E1 | Final | my real takes |
|---|---|---|---|
| similarity to reference, en: mean / std / min (3 seeds) | 0.502 / 0.068 / 0.324 | 0.730 / 0.034 / 0.618 | – |
| same voice across languages: en-hi / en-Hinglish / hi-Hinglish | 0.797 / 0.863 / 0.893 | 0.962 / 0.964 / 0.990 | 0.869 / 0.958 / 0.953 |
| speaking rate, words/s | 2.38 | 2.54 | 2.52 |
| median pause in comma sentences, ms | 250 | 245 | 300 |
| statements ending with a fall | 29/41 | 50/58 | 150/185 |
| yes/no questions ending with a rise | 5/10 | 4/9 | 6/11 |
| listeners: more natural (16 pairs) | 4 | 11 (1 same) | – |

Final is steadier (half the spread across sentences) and nearly one voice across languages, partly because its Hinglish also uses the Hindi path. Its rate and falling statements are close to my real takes. Neither model reliably raises yes/no questions, and my own takes only rise on 6 of 11, so the training data may not teach it. Endings are a 300 ms pyin pitch slope; unvoiced endings are left out.



## 7. Failure analysis

**Hinglish (fixed).** Pretrained XTTS sends Roman Hinglish through its English front-end and garbles it ("Aapka refund teen din mein..." came out as babble in an early probe). I tested the cheapest fix first: write the same sentence in Devanagari and send it as Hindi. The 2×2 below separates the two effects (3 seeds; [hinglish](../results/bootstrap_E3_vs_E3_deva.txt), [held-out](../results/bootstrap_E3_heldout_en_vs_E3_heldout_deva.txt), [summary](../results/summary.md)):

| Hinglish CER | Roman, sent as en | Devanagari, sent as hi |
|---|---|---|
| E1 pretrained, benchmark (16) | 0.400 | 0.097 |
| E3 fine-tuned, benchmark (16) | 0.222 | 0.098 |
| E1, held-out (10) | 0.348 | 0.128 |
| E3, held-out (10) | 0.205 | 0.119 |

The route gives the intelligibility; fine-tuning gives the voice (with the route fixed, Hinglish similarity 0.676 → 0.738). On E3, the route is worth −0.124 [−0.177, −0.068] on the benchmark and −0.086 [−0.146, −0.034] on the held-out sentences. The held-out check matters because I chose the lexicon's function words while looking at the 16 benchmark rows. Training on Devanagari Hinglish instead (E4) added nothing. What IndicXlit still gets wrong is mostly loanwords, for example "Pune" → प्यून.

**Names (not fixed, main weakness).** E3 misses 0.45 of name words, against 0.02 of ordinary English words ([error_analysis](../results/error_analysis_E3_run1.md)): "Anjali Reddy" → "and daddy ready", "Guwahati" → "Buahtai", and one babble after "Venkatesh Iyer". Seen names are no easier: E3 misses 42% of seen and 41% of unseen name words, the pretrained model 25% and 30% ([names_seen](../results/names_seen.md)). On English name rows fine-tuning made CER worse, 0.030 → 0.106, +0.076 [+0.005, +0.193] (8 rows; the interval only just excludes 0, one row is the babble). Whisper itself misses 13 of 48 name words (0.27) on my real takes ([names_real](../results/names_real.txt)), about the pretrained model's rate, so E3's extra name errors are the model's. Numbers and dates the listeners marked wrong had correct normaliser text, so those are delivery ("E M I" heard as "AMI").

## 8. Production

Measured on a T4 for the final pipeline over 176 sentences (the benchmark twice; [latency](../results/runs/extras/latency/profile.json)):

| | value |
|---|---|
| model on disk | 1789 MB |
| peak GPU memory | 2.47 GB |
| load model (once) | 30 s |
| voice conditioning (once per voice, cacheable) | 0.89 s |
| normaliser | 0.21 ms per sentence |
| Hinglish transliteration | 140 ms median per sentence on Kaggle's CPU |
| time to first audio (streaming, chunk 20) | 432 ms median, 456 ms p95 |
| real-time factor | 0.51 median, 0.58 p95 |

**Time to first audio for Hinglish.** The 432 ms above is measured from the start of synthesis. Hinglish transliteration runs before that, so end-to-end first audio for Hinglish is about 432 + 140 ≈ 570 ms. It isn't model load or a slow first call (both timed separately; per-pass medians 143 and 138 ms, [profile](../results/runs/extras/latency/profile.csv)). It is IndicXlit's beam-search decoding on Kaggle's slow CPU: 20 ms per sentence on my Mac ([translit_timing](../results/translit_timing.json)). Greedy decoding is only about 30% faster and changes 5 of 26 outputs, so I kept beam search and added a per-word cache: same output, 0.02 ms for a repeated sentence. New words still cost the full time, so new Hinglish text is ~0.57 s on that CPU; IndicXlit on GPU is untested.

**Streaming and bottlenecks.** Audio streams in chunks, so first audio arrives after about 0.43 s. The GPT's token-by-token decoding is the bottleneck (RTF 0.51); memory isn't (2.5 GB per copy).

**Hundreds of concurrent calls.** The agent only talks for part of each call. Assume it speaks 40% of the time (an assumption, not measured). One unbatched stream at RTF 0.51 makes 1 / 0.51 ≈ 2.0 seconds of audio per second, so one T4 covers about 2.0 / 0.4 ≈ 5 calls if their speaking turns are spread out, and 100 calls need about 100 / 5 = 20 T4s. Turns collide, so p95 first-audio time would get worse before that point. That is still too expensive, so the plan is: continuous batching of many calls' GPT decoding on one GPU (most of the gain); cached voice conditioning; fp16 inference (untested; T4s run it much faster, quality needs checking); the text front-end on CPU workers with the word cache; pre-rendered fixed phrases; babble guards (token limit, duration check); and autoscaling on p95 first-audio time.

Batched throughput isn't measured. **Licence:** XTTS-v2's weights are non-commercial (Coqui Public Model License), and Coqui shut down in January 2024 ([coqui-ai/TTS discussion #4145](https://github.com/coqui-ai/TTS/discussions/4145)), so there is no one to buy a commercial licence from. A real deployment needs a different base model; the pipeline (data, normaliser, Hinglish route, evaluation) carries over.

## 9. Limits and next steps

**Limits:** 24 minutes of one speaker; 12 Hindi rows; Hinglish CER is rough; Whisper misreads names; one consensus listening session. **Next:** name and place recordings plus a respelling list checked on unseen names; a third, mostly Hindi session; independent listeners; a commercially licensed base model; measured batching, fp16 and IndicXlit on GPU.
