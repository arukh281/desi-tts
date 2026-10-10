# desi-tts: fine-tuning XTTS-v2 on my own voice for Indian English, Hindi and Hinglish

Aradhya Khandelwal, October 2026. Code and writing drafted with AI assistance; recordings, decisions and listening tests are mine.

Every number below comes from a file in [`results/`](../results); the file is named next to each table. The experiment table is in [EXPERIMENTS.md](EXPERIMENTS.md) and audio samples are on the [sample page](https://arukh281.github.io/desi-tts/).

## 1. Summary

I fine-tuned the GPT part of XTTS-v2 on 24 minutes of my own speech and added two text front-ends: a normaliser for numbers, currency, dates and acronyms, and a route that sends Hinglish to the model in Devanagari. On an 88-sentence benchmark, averaged over 3 seeds, the final system has a character error rate (CER) of 0.061 against 0.125 for the pretrained model with the same normaliser (difference −0.064, 95% interval [−0.110, −0.025]). Almost all of that gain is Hinglish (0.400 → 0.098). English and Hindi CER don't change beyond noise. Speaker similarity to my voice rises from 0.50 to 0.73 in English, against 0.76 for my real recordings. In one joint listening session, three listeners picked the fine-tuned model as more like me in 15 of 16 pairs. Names are the main weakness left, and fine-tuning made them slightly worse.

## 2. Data

**Why my own voice.** I needed one clean speaker who covers Indian English, Hinglish and Hindi. None of the open sets I checked (IndicTTS, SYSPIN, Rasa, LIMMITS, Svarah) has one speaker across all three, and mixing in another speaker's Hindi would blur the voice. Recording myself also means I know the consent and licence.

**Recording.** MacBook Air built-in mic, 48 kHz, two sessions (8 and 10 Oct; 16.4 and 7.8 minutes used), with a small recorder that shows a level meter and warns on clipping, low level and long takes. I planned about an hour and three sessions; I stopped after two, so the set is **24 minutes** ([dataset_card](../results/dataset_card.md)).

| group | sent to XTTS as | clips | minutes | train / val |
|---|---|---|---|---|
| Indian English | en | 154 | 12 | 146 / 8 |
| Hinglish (Roman script) | en | 82 | 5.4 | 78 / 4 |
| Hindi (Devanagari) | hi | 97 | 6.6 | 92 / 5 |
| total | | 333 | 24 | 316 / 17 |

**Cleaning.** Each take is checked for clipping, noise floor and SNR (median 37 dB), long pauses and length; edge clicks from the Enter key are cut; silence is trimmed; loudness is normalised to −23 LUFS and audio resampled to 22.05 kHz. Whisper large-v3 then transcribes every take and its CER against the prompt flags misreads. Two lessons came out of this. First, Whisper forced to English *translates* Hinglish, so Hinglish CER from that check means nothing and it is report-only. Second, every automatic drop I listened to except one was Whisper mishearing a correct take (for example, "twelve thousand" heard as 1200), so drops need a listen before they count. Of 334 takes, 333 are used: the one real drop (p0293) was cut off. 45 takes are flagged but kept (reasons: 35 borderline transcripts, 6 low SNR, 5 long pauses; one take has two).

**Provenance.** The 650 prompts were written with AI help: p0001-p0480 by a template script, p0481-p0650 drafted by an AI assistant as a Hindi/Hinglish top-up, and 24 prompts changed to masculine first person on my instruction. I read Hindi from an AI-written romanisation shown by the recorder, while the saved transcript stays in Devanagari. The 88 benchmark sentences and 3 reference sentences are kept out of training, which a leakage check confirms. The audio and the prompt list stay private.

## 3. Model and text

**Model.** XTTS-v2 (coqui-tts 0.27.5). It already speaks English and Hindi, clones a voice from short reference audio and can stream. I fine-tune only its GPT part: 441 M of 518 M parameters ([training_summary](../results/runs/E3_run2/train/train_summary.json)). The GPT decides which sounds to make, so it carries accent and pronunciation. The audio codec and the HiFi-GAN decoder stay frozen, which keeps training within a 16 GB T4. XTTS-v2's licence is non-commercial, which matters for production (section 9).

**Text representation.** XTTS reads characters, not phonemes. That turned out to help: wrong readings can often be fixed in the text before the model sees it.
- **Normaliser** (`text/normalize.py`): ₹, numbers, lakh/crore, dates, times and acronyms become the words a person would say ("₹12,750" → "twelve thousand seven hundred and fifty rupees", "KYC" → "K Y C"). It reproduces all 64 expected spoken forms in the benchmark. Without it, pretrained XTTS crashes on 8 of 12 Hindi rows, because its own digit expander fails on Hindi.
- **Hinglish route** (`text/translit.py`): Roman Hinglish is turned into Devanagari with IndicXlit (AI4Bharat, MIT licence) plus a lexicon of 178 number and everyday words, then sent to XTTS as Hindi. Section 7 covers why.

## 4. Training

Final run E3: sessions 1+2 (316 train / 17 val clips), learning rate 5e-6 (the coqui recipe default, low so the model keeps its English and Hindi), batch 2 with gradient accumulation 8 (16 clips per update), a 2000-step target (2054 steps run, 13 epochs), fp32, seed 1234. Weights are saved at 500, 1000, 1500 and 2000 steps. The checkpoint is chosen on the 17 val clips only: lowest val loss, among checkpoints whose val CER is within 0.05 of the best and that have no more than 2 extra duration flags. That picked **step 1500**. Step 2000 had the lowest loss but a val CER of 0.135, a sign of babbling ([selection](../results/runs/E3_run2/select/selection.csv)).

**Time.** 2054 steps (median 0.44 s per step), **16.4 minutes** of summed step time on one T4, peak 9.4 GB VRAM ([training](../results/training.md)). The whole Kaggle job, including setup, checkpoint selection and benchmark synthesis, took about 50 minutes ([run_info](../results/runs/E3_run2/run_info.json)). I trained E3 twice with the same seed; run 2 gave exactly the same selection table as run 1, so the pick is reproducible ([run 1](../results/runs/E3_run1/select/selection.csv), [run 2](../results/runs/E3_run2/select/selection.csv)).

## 5. Benchmark and experiments

**Benchmark.** 88 sentences, drafted with AI, reviewed by me and frozen (commit dc343ba, 7 Oct) before any training: 60 Indian English in 9 categories (conversational, names, places, numbers, dates and times, currency, acronyms, questions, long), 16 Hinglish and 12 Hindi. Names and places are tagged *seen* (in my prompts) or *unseen*. The benchmark is used only for final comparisons. Every choice (checkpoint, sampling, splitting, E4) was made on val clips or on two held-out sets: 10 extra Hinglish sentences and 5 long sentences.

**Metrics.**
- *Intelligibility:* Whisper large-v3 CER against the spoken form. Hinglish is scored with Hindi Whisper against AI-drafted Devanagari references, because English Whisper translates it. That makes Hinglish CER a rough signal.
- *Voice:* ECAPA speaker similarity to my reference clip in the same language. Similarity drops across languages even for one stock voice, so I only compare within a language.
- *Delivery:* duration checks for stretched, rushed or babbling clips.
- *Noise:* final numbers average 3 seeds. A paired bootstrap over sentences (10,000 resamples) gives a 95% interval, and I call a difference real only if the interval excludes 0. A crashed sentence counts as CER 1.0, so a system can't look better by skipping what it can't say.

**Experiments** (details in [EXPERIMENTS.md](EXPERIMENTS.md)):
- **E0** is the pretrained model on raw text; it crashes on Hindi digits. **E1** is the pretrained model with the normaliser, and is the baseline.
- **E2** fine-tuned on session 1 and picked the checkpoint by val CER. It **failed**: it chose step 250, which sounded robotic and not like me. CER on 8 val clips was too noisy to choose by, so I switched to the val-loss rule above.
- **E2b** repeats E3's recipe on session 1 only, to see what the extra 8 minutes bought. I trained it twice. Both runs picked step 1000, after which val loss rises: 16 minutes overfits sooner. Run 2 didn't reproduce run 1 bit for bit (val loss at the picked step within 0.002, val CER moved), unlike E3. Run 1, single seed: CER en 0.028, hi 0.062, Hinglish 0.263. Run 2, 3 seeds, against E3: E3 − E2b = +0.002 [−0.016, +0.022] overall, +0.008 [−0.015, +0.034] English, +0.035 [−0.003, +0.094] Hindi, and −0.045 [−0.076, −0.017] Hinglish ([bootstrap_E2b_vs_E3](../results/bootstrap_E2b_vs_E3.txt)). So the extra 8 minutes measurably helped only Hinglish, and slightly the voice (English similarity 0.721 → 0.730). It also shows E2's failure came from the checkpoint pick, not from having less data.
- **E4** trains Hinglish as Devanagari, **E5** sweeps sampling settings and **E6** splits long sentences. None of them beat noise, so none were adopted.

## 6. Results

3 seeds, normaliser on, my 3 reference clips ([summary](../results/summary.md), bootstraps in [results/](../results)).

| | E1 pretrained | E3 fine-tuned | **Final** (E3 + Hinglish route) | my real voice (val clips) |
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

Fine-tuning alone (E3 − E1) is −0.041 [−0.083, −0.009] overall and −0.178 [−0.247, −0.118] on Hinglish. By category, E1 → Final: questions 0.291 → 0.026, currency 0.103 → 0.020, dates and times 0.189 → 0.093, numbers 0.169 → 0.105, but names only 0.107 → 0.095, and places 0.073 → 0.091 ([bootstrap_E1_vs_Final](../results/bootstrap_E1_vs_Final.txt)).

How to read the real-voice row: it is my own val recordings scored with the same Whisper and ECAPA settings. It is a ceiling for similarity and a floor for CER, but it uses different sentences and only 17 clips. English synthetic speech (0.730) is close to my real voice (0.763). Hindi synthetic is *above* real (0.706 vs 0.615), probably because synthesis is conditioned on the reference clip, so it can sound more like that clip than my other takes do. Real English CER (0.100) being worse than synthetic is a small-sample oddity: 8 clips with names and numbers.

**Listening test.** One joint session: three people listened together and agreed each answer, so it is one consensus judgement, not three independent ones. The page was blind and in random order: 22 A/B pairs, 16 E1 vs E3, 4 Hinglish Roman vs Devanagari on E3, and 2 repeats with A and B swapped. The E3 audio was from E3 run 1, whose training matches run 2 exactly ([listening](../results/listening_joint_session1.md)).

| question | result |
|---|---|
| more like Aradhya, E1 vs E3 (16) | E3 15, same 1, E1 0 |
| more natural, E1 vs E3 (16) | E3 11, E1 4, same 1 |
| a word wrong or unclear | E1 11/16, E3 9/16 |
| Hinglish, Roman vs Devanagari (4) | Devanagari 4/4 on both questions; wrong word Roman 4/4, Devanagari 1/4 |
| repeated pairs answered the same way | 2/2 |

The voice change is clear. Both systems still get words wrong often, which matches the names result below.

## 7. Failure analysis

**Hinglish (fixed).** Pretrained XTTS sends Roman Hinglish through its English front-end and garbles it ("Aapka refund teen din mein..." came out as babble in an early probe). I tested the cheapest fix first: write the same sentence in Devanagari and send it as Hindi. The 2×2 below separates the two effects (3 seeds; [hinglish](../results/bootstrap_E3_vs_E3_deva.txt), [held-out](../results/bootstrap_E3_heldout_en_vs_E3_heldout_deva.txt), [summary](../results/summary.md)):

| Hinglish CER | Roman, sent as en | Devanagari, sent as hi |
|---|---|---|
| E1 pretrained, benchmark (16) | 0.400 | 0.097 |
| E3 fine-tuned, benchmark (16) | 0.222 | 0.098 |
| E1, held-out (10) | 0.348 | 0.128 |
| E3, held-out (10) | 0.205 | 0.119 |

The route gives the intelligibility; fine-tuning gives the voice (with the route fixed, Hinglish similarity 0.676 → 0.738). On E3, the route is worth −0.124 [−0.177, −0.068] on the benchmark and −0.086 [−0.146, −0.034] on the held-out sentences. The held-out check matters because I chose the lexicon's function words while looking at the 16 benchmark rows. Training on Devanagari Hinglish instead (E4) added nothing. What IndicXlit still gets wrong is mostly loanwords, for example "Pune" → प्यून.

**Names (not fixed, main weakness).** By word type, E3 misses 0.45 of name words, against 0.02 of ordinary English words ([error_analysis](../results/error_analysis_E3_run1.md)). Examples: "Anjali Reddy" → "and daddy ready", "Guwahati" → "Buahtai", and one babble where the model added a sentence after "Venkatesh Iyer". Pooled over 3 seeds, E3 misses 42% of seen name words and 41% of unseen ones; the pretrained model misses 25% and 30% ([names_seen](../results/names_seen.md)). So having a name in training didn't help, and fine-tuning on 24 minutes made names a little worse. Part of this is Whisper, which also struggles with Indian names, so the true rate is lower than measured. But the listeners marked name errors too. The numbers and dates the listeners marked wrong had correct normaliser text, so those come from the model's delivery, for example "E M I" heard as "AMI".

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

**Time to first audio for Hinglish.** The 432 ms above is measured from the start of synthesis. Hinglish transliteration runs before that, so end-to-end first audio for Hinglish is about 432 + 140 ≈ 570 ms. This isn't model loading or a slow first call: both are done and timed before the loop, and the per-pass medians are 143 and 138 ms ([profile](../results/runs/extras/latency/profile.csv)). It is CPU decoding of IndicXlit (beam search) on Kaggle's slow CPU; the same code takes 20 ms per sentence on my Mac ([translit_timing](../results/translit_timing.json)). Greedy decoding is only about 30% faster and changes 5 of 26 outputs ([translit_timing](../results/translit_timing.json)), so I didn't switch. I added a per-word cache instead: output is identical, and a repeated sentence takes 0.02 ms. New words still cost the full time, so on a slow CPU the honest number for new Hinglish text is ~0.57 s. Running IndicXlit on the GPU or a faster CPU should close most of the gap, but I haven't measured that.

**Streaming and bottlenecks.** XTTS streams audio from its GPT in chunks, so first audio arrives after about 0.43 s while the rest is generated. The GPT's token-by-token decoding is the bottleneck: one stream uses about half a T4 in real time (RTF 0.51). Memory isn't the limit, at 2.5 GB per model copy.

**Hundreds of concurrent calls.** At RTF 0.5, one unbatched stream per GPU keeps up with about two live calls, so a hundred calls would need about 50 T4s. That is too expensive, so the real plan is:
- batch many calls' GPT decoding on one GPU (continuous batching), which is where most of the gain is;
- cache each voice's conditioning once;
- run the text front-end on CPU workers, with the word cache;
- pre-render fixed phrases such as greetings and hold messages;
- guard against babbling with a token limit and a duration check;
- scale on p95 first-audio time.

I haven't measured batched throughput, so I can't give a calls-per-GPU number yet. XTTS-v2's non-commercial licence would also have to be dealt with before a real deployment.

## 9. Limits and next steps

- **Small data and small tests.** 24 minutes of one speaker. Hindi has 12 benchmark rows and its interval is wide; Hinglish CER is a rough Devanagari score. The listening test is one consensus session of three people, not independent raters.
- **Metric limits.** Whisper misreads Indian names and spells Hindi loanwords inconsistently, so some measured errors are the ASR's.
- **Names.** The next step is targeted recordings of names and places plus a respelling list, checked on unseen names.
- **More data.** A third session, mostly Hindi and names, and independent listeners, each rating separately.
- **Production.** Measure batched throughput and IndicXlit on GPU; consider a base model with a commercial licence.
