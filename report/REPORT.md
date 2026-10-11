# desi-tts: fine-tuning XTTS-v2 on my own voice for Indian English, Hindi and Hinglish

Aradhya Khandelwal, October 2026. I recorded the data and made the final calls. The code and most of the writing were done with a lot of AI help (Claude).

Every number comes from a file in [results/](../results). Files: [EXPERIMENTS.md](EXPERIMENTS.md), [sample page](https://arukh281.github.io/desi-tts/)

## 1. Summary

The fine-tuned voice sounds like me and reads Hinglish much better, but it still gets Indian names wrong.

I fine-tuned XTTS-v2's GPT on 24 minutes of my speech and added two text steps: one writes numbers, money, dates and acronyms as words, and one sends Hinglish in Devanagari. On an 88-sentence benchmark (3 seeds), the character error rate (CER) drops from 0.125 for the pretrained model to 0.061. That is a real difference, and almost all of it is Hinglish (0.400 down to 0.098). English and Hindi stay within noise. English speaker similarity goes from 0.50 to 0.73 (my real recordings: 0.76). In one joint listening session, where three people agreed each answer together, they picked the fine-tuned model as more like me in 15 of 16 pairs. On the 8 English name rows the fine-tuned model is worse (0.030 up to 0.106).

Files: [summary](../results/summary.md), [bootstrap_E1_vs_Final](../results/bootstrap_E1_vs_Final.txt), [listening](../results/listening_joint_session1.md)

## 2. Data

I recorded 24 minutes of my own voice in two sessions and used 333 of 334 takes.

None of the open sets I checked (IndicTTS, SYSPIN, Rasa, LIMMITS, Svarah) has one clean speaker across all three languages, and mixing speakers would blur the voice. I recorded on the MacBook Air mic at 48 kHz with a recorder that warns about clipping and level. I planned about an hour over three sessions and stopped after two (8 and 10 Oct, 16.4 and 7.8 minutes used).

| group | sent to XTTS as | clips | minutes | train / val |
|---|---|---|---|---|
| Indian English | en | 154 | 12 | 146 / 8 |
| Hinglish (Roman script) | en | 82 | 5.4 | 78 / 4 |
| Hindi (Devanagari) | hi | 97 | 6.6 | 92 / 5 |
| total | | 333 | 24 | 316 / 17 |

Each prompt is its own take, so nothing needed cutting or aligning. Takes are checked for clipping, SNR (median 37 dB), pauses and length, trimmed, set to −23 LUFS and resampled to 22.05 kHz. The transcript is the prompt as written, already in spoken form, with punctuation and case kept for phrasing; XTTS's tokenizer does its usual cleaning. Whisper large-v3 checks every take. It translates Hinglish, so that score is only for reference, and every automatic drop but one was Whisper mishearing a good take, so I listen before dropping. The one real drop (p0293) was cut off. 45 takes are flagged but kept (35 borderline transcripts, 6 low SNR, 5 long pauses; one has two).

The 650 prompts were written with AI help (a template script for p0001-p0480, an AI-drafted Hindi and Hinglish top-up for p0481-p0650, and 24 changed to masculine first person at my request). I read Hindi from an AI-written romanisation; transcripts stay in Devanagari. A leakage check keeps benchmark and reference sentences out of training. Audio and prompts stay private.

Files: [dataset_card](../results/dataset_card.md)

## 3. Model and text

I fine-tune only the GPT part of XTTS-v2, and I fix a lot of reading mistakes in the text before it reaches the model.

XTTS-v2 (coqui-tts 0.27.5) already speaks English and Hindi, copies a voice from short clips, and streams. Its GPT is trained to predict the discrete acoustic tokens of XTTS's codec (a discrete VAE), and the HiFi-GAN decoder turns the GPT's latent states into audio. The GPT carries the accent. I train only it (441 M of 518 M parameters) and keep the codec and HiFi-GAN frozen, which fits a 16 GB T4.

XTTS reads plain letters, split into subword pieces by a BPE tokenizer, so I could fix a lot of mistakes just by changing the text before it goes in.

- The normaliser (`text/normalize.py`) writes ₹, numbers, lakh and crore, dates, times and acronyms the way a person says them: "₹12,750" becomes "twelve thousand seven hundred and fifty rupees" and "KYC" becomes "K Y C". It gives all 64 expected spoken forms in the benchmark. Without it the pretrained model crashes on 8 of 12 Hindi rows, because its own number expander fails on Hindi.
- The Hinglish step (`text/translit.py`) turns Roman Hinglish into Devanagari with IndicXlit (AI4Bharat, MIT licence) and a list of 178 number and everyday words, then sends it as Hindi (section 7).

Files: [train_summary](../results/runs/E3_run2/train/train_summary.json)

## 4. Training

Training took 16 minutes on one T4, and I picked the checkpoint using only the 17 validation clips.

E3 uses both sessions, learning rate 5e-6 (the recipe default, low to keep the base model's English and Hindi), batch 2, gradient accumulation 8, fp32, seed 1234. A step is one batch of 2 clips, so the 2054 steps I ran (target 2000, 13 epochs) are about 257 weight updates. From the weights kept at 500, 1000, 1500 and 2000 steps, I took the lowest validation loss among those within 0.05 of the best validation CER and with at most 2 extra duration flags. That picked step 1500. Step 2000 had the lowest loss but a validation CER of 0.135, a sign of babbling. The 2054 steps took 16.4 minutes in total (median 0.44 s per step, peak 9.4 GB). Training E3 a second time with the same seed gave exactly the same selection table.

Files: [run_config](../results/runs/E3_run2/train/run_config.json), [selection run 2](../results/runs/E3_run2/select/selection.csv), [run 1](../results/runs/E3_run1/select/selection.csv), [training](../results/training.md)

## 5. Benchmark and experiments

All choices were made on validation clips or held-out sets. The benchmark was used only for the final comparison.

The benchmark has 88 sentences, drafted with AI, reviewed by me and frozen before any training (commit dc343ba, 7 Oct): 60 Indian English in 9 categories, 16 Hinglish and 12 Hindi. Names and places are tagged seen (in my prompts) or unseen. The held-out sets are 10 Hinglish, 5 long and 20 name sentences.

How I measure. Intelligibility: Whisper large-v3 CER. Hinglish uses Hindi Whisper against AI-drafted Devanagari references, so it is rough. Voice and delivery: ECAPA similarity to my reference clip in the same language, and duration checks for stretched or babbling clips. Noise: 3 seeds and a paired bootstrap over sentences (10,000 resamples). A difference is real only if the 95% interval leaves out 0. A crash counts as CER 1.0.

The experiments in short (details in EXPERIMENTS.md). E0, the pretrained model on raw text, crashes on Hindi digits. E1 adds the normaliser and is the baseline. E2 (session 1, checkpoint picked by validation CER) failed: step 250 sounded robotic, though its English similarity was already 0.718. So ECAPA similarity says who is speaking, and little about naturalness. I switched to the validation-loss rule. E2b repeats E3's recipe on session 1 only, twice. The extra 8 minutes in E3 made a real difference only for Hinglish. E4 (training Hinglish as Devanagari), E5 (sampling settings) and E6 (splitting long sentences) didn't beat noise, so I kept none of them.

Files: [EXPERIMENTS.md](EXPERIMENTS.md), [bootstrap_E2b_vs_E3](../results/bootstrap_E2b_vs_E3.txt)

## 6. Results

Fine-tuning plus the Hinglish step gives a real drop in errors, almost all on Hinglish, and a much closer voice (3 seeds, normaliser on, my 3 reference clips).

| | E1 pretrained | E3 fine-tuned | Final (E3 + Hinglish step) | my real voice (val clips) |
|---|---|---|---|---|
| CER, all 88 | 0.125 | 0.084 | 0.061 | – |
| CER, English (60) | 0.037 | 0.043 | 0.043 | 0.100 (8 clips) |
| CER, Hindi (12) | 0.198 | 0.102 | 0.102 | 0.061 (5 clips) |
| CER, Hinglish (16, rough) | 0.400 | 0.222 | 0.098 | not comparable* |
| speaker sim en / hi / Hinglish | 0.502 / 0.643 / 0.589 | 0.730 / 0.706 / 0.720 | 0.730 / 0.706 / 0.738 | 0.763 / 0.615 / 0.729 |

\*Scored with English Whisper (0.521), which translates it. Final's Hinglish similarity is the Devanagari step's.

| change in CER | E3 − E1 | real? | Final − E1 | real? |
|---|---|---|---|---|
| all | −0.041 [−0.083, −0.009] | yes | −0.064 [−0.110, −0.025] | yes |
| English | +0.006 [−0.014, +0.029] | no | +0.006 [−0.014, +0.029] | no |
| Hindi | −0.096 [−0.331, +0.054] | no | −0.096 [−0.331, +0.054] | no |
| Hinglish | −0.178 [−0.247, −0.118] | yes | −0.303 [−0.392, −0.215] | yes |

By category, E1 to Final: questions 0.291 to 0.026, currency 0.103 to 0.020, dates 0.189 to 0.093, numbers 0.169 to 0.105, places 0.073 to 0.091, names 0.107 to 0.095. The names category has 13 rows (8 English, 3 Hinglish, 2 Hindi). Its small drop comes from the 3 Hinglish rows the Devanagari step fixes, while the English and Hindi name rows get worse (section 7).

The real-voice column is my own validation recordings, scored the same way. These are different sentences and only 17 clips, so treat it as a rough guide; Hindi synthetic similarity is even above it (0.706 against 0.615).

The listening test was one joint session: three people listened together and agreed each answer, so it counts as one judgement. I ran the session and played the clips but didn't answer. I also listened to every benchmark output myself during error analysis. The page was blind and shuffled: 16 E1 against E3 pairs (E3 audio from run 1), 4 Hinglish Roman against Devanagari, and 2 swapped repeats. More like Aradhya: E3 15 of 16, 1 same. More natural: E3 11, E1 4, 1 same. A word wrong: E1 11 of 16, E3 9 of 16. Hinglish: Devanagari won 4 of 4, with a wrong word in 4 of 4 Roman clips and 1 of 4 Devanagari. Both repeats matched. The final voice is also steadier and speaks at about my rate:

| | E1 | Final | my real takes |
|---|---|---|---|
| similarity to reference, en: mean / std / min (3 seeds) | 0.502 / 0.068 / 0.324 | 0.730 / 0.034 / 0.618 | – |
| same voice across languages: en-hi / en-Hinglish / hi-Hinglish (seed 1234) | 0.797 / 0.863 / 0.893 | 0.962 / 0.964 / 0.990 | 0.869 / 0.958 / 0.953 |
| speaking rate, words/s (seed 1234) | 2.38 | 2.54 | 2.52 |
| median pause in comma sentences, ms (seed 1234) | 250 | 245 | 300 |
| statements ending with a fall (seed 1234) | 29/41 | 50/58 | 150/185 |
| yes/no questions ending with a rise (seed 1234) | 5/10 | 4/9 | 6/11 |
| listeners: more natural (16 pairs) | 4 | 11 (1 same) | – |

Counts differ because my real takes are 333 other sentences and unvoiced endings are left out (more of them in E1). Final's spread is half of E1's, and its languages sound more alike than mine. I don't count that as better: every synthetic sentence uses the same reference clips and Hinglish goes through the Hindi path, while real speech changes with the language. Neither model reliably raises yes/no questions, and my own takes rise on only 6 of 11.

Files: [summary](../results/summary.md), [bootstrap_E1_vs_E3](../results/bootstrap_E1_vs_E3.txt), [bootstrap_E1_vs_Final](../results/bootstrap_E1_vs_Final.txt), [listening](../results/listening_joint_session1.md), [consistency](../results/consistency.md), [prosody](../results/prosody.md)

## 7. Failure analysis

Sending Hinglish as Devanagari fixed Hinglish. Names are still the main problem; my fixes didn't work.

Pretrained XTTS reads Roman Hinglish through its English path and garbles it. The cheapest fix was to send the same sentence in Devanagari as Hindi (3 seeds):

| Hinglish CER | Roman, as en | Devanagari, as hi | Devanagari − Roman |
|---|---|---|---|
| E1 pretrained, benchmark (16) | 0.400 | 0.097 | |
| E3 fine-tuned, benchmark (16) | 0.222 | 0.098 | −0.124 [−0.177, −0.068] |
| E1, held-out (10) | 0.348 | 0.128 | |
| E3, held-out (10) | 0.205 | 0.119 | −0.086 [−0.146, −0.034] |

The Devanagari step gives the intelligibility and fine-tuning gives the voice: with the step held fixed, Hinglish similarity goes from 0.676 to 0.738. On E3 the step is a real difference on both sets.

E3 misses 0.45 of name words against 0.02 of ordinary English words. "Anjali Reddy" was heard as "and daddy ready", "Guwahati" as "Buahtai", and in one case (b009, CER 1.25) the model babbled on after "Venkatesh Iyer". Names from my training prompts are no easier (42% of seen and 41% of unseen name words missed).

| name rows, CER | n | E1 | Final | Final − E1 | real? |
|---|---|---|---|---|---|
| English | 8 | 0.030 | 0.106 | +0.076 [+0.005, +0.193] | yes, barely |
| Hinglish | 3 | 0.355 | 0.084 | −0.272 [−0.411, −0.092] | yes |
| Hindi | 2 | 0.041 | 0.065 | +0.024 [+0.017, +0.031] | yes (2 rows) |
| all names | 13 | 0.107 | 0.095 | −0.012 [−0.115, +0.094] | no |

One of the 8 English rows is the babble. Some error is Whisper's: on my real takes it misses 13 of 48 name words (0.27), about the pretrained model's rate, so E3's extra name errors come from the model. On 20 new held-out name sentences (AI-drafted, like the benchmark), a lower temperature, a respelling list and a fine-tune with name clips repeated 3 times all failed. Final misses 0.34 of name words and they miss 0.37 to 0.46. I wrote the respelling list before any audio was generated, and it covers these names. A babble guard never triggered on this set, so it is untested.

Files: [bootstrap_E3_vs_E3_deva](../results/bootstrap_E3_vs_E3_deva.txt), [held-out](../results/bootstrap_E3_heldout_en_vs_E3_heldout_deva.txt), [error_analysis](../results/error_analysis_E3_run1.md), [E3 run 1](../results/runs/E3_run1/bench_norm_on/asr_eval.csv), [names_seen](../results/names_seen.md), [bootstrap_names](../results/bootstrap_names_E1_vs_Final.txt), [names_real](../results/names_real.txt), [EXPERIMENTS.md](EXPERIMENTS.md) (E7, E8)

## 8. Production, limits and next steps

First audio comes in under half a second on a T4, but a real deployment needs batching and a different base model.

Final pipeline on a T4, benchmark run twice (176 sentences):

| | value |
|---|---|
| model on disk / peak GPU memory | 1789 MB / 2.47 GB |
| load model (once) / voice conditioning (once per voice) | 30 s / 0.89 s |
| normaliser / Hinglish transliteration (Kaggle CPU), per sentence | 0.21 ms / 140 ms median |
| time to first audio (streaming, chunk 20) | 432 ms median, 456 ms p95 |
| real-time factor | 0.51 median, 0.58 p95 |

Hinglish starts later, at about 570 ms (432 + 140), because transliteration runs first. Loading is timed separately and both passes give 143 and 138 ms, so the cause is IndicXlit's beam search on Kaggle's slow CPU (my Mac: 20 ms). Greedy decoding was only about 30% faster and changed 5 of 26 outputs, so I added a word cache instead: same output, 0.02 ms for a repeated sentence, full cost for new words.

The bottleneck is the GPT's token-by-token decoding. For many calls at once, I assume the agent talks 40% of each call (not measured). One unbatched stream makes about 2.0 s of audio per second (1 / 0.51), so a T4 covers about 5 calls (2.0 / 0.4) if turns are spread out, and 100 calls need about 20 T4s (100 / 5). That is too expensive, so the plan is batching many calls on one GPU (most of the gain; I haven't measured it), cached voice conditioning, fp16 (untested), text steps on CPU workers, pre-rendered phrases, babble guards and scaling on p95 first-audio time.

XTTS-v2's weights are non-commercial and Coqui shut down in January 2024, so a real deployment needs another base model; the rest carries over. Limits: 24 minutes, one speaker, 12 Hindi rows, a rough Hinglish score, one listening session. Next: name recordings, routing Indian names through Devanagari word by word, more Hindi, independent listeners, batching and fp16.

Files: [latency](../results/runs/extras/latency/profile.json), [profile](../results/runs/extras/latency/profile.csv), [translit_timing](../results/translit_timing.json), [coqui-ai/TTS discussion #4145](https://github.com/coqui-ai/TTS/discussions/4145)
