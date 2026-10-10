# Days

Short log of what I did each day and why, including what I chose not to do.

## Day 1 (5 Oct): decisions, no training yet

**Compute: free Kaggle T4 (16 GB).** No budget, and the brief says more compute doesn't score higher. Because it's smaller than the 24–48 GB the brief assumes, I'll train only part of the model with a small batch, and save checkpoints often since Kaggle sessions can be cut.

**Model: XTTS-v2, fine-tuning only the GPT part.** It already speaks English and Hindi, clones a voice and can stream. The GPT part turns text into sound, so it carries the accent; the rest stays frozen to fit in 16 GB. It reads characters, not phonemes, so wrong pronunciations can often be fixed in the text. Not chosen: Indic Parler-TTS (already strong, harder to fine-tune on a T4) or training from scratch. Note: XTTS-v2's licence is non-commercial.

**Data: my own voice only.** None of the open sets I checked (IndicTTS, SYSPIN, Rasa, LIMMITS, Svarah) has one clean speaker covering Indian English, Hinglish and Hindi. Mixing in another speaker's Hindi would blur the voice. Hindi will have the least data, so I expect it to be the weak spot.

**Recording script: 480 sentences** (English, Hinglish, Hindi), mostly banking and support. Numbers and acronyms are written out as spoken ("K Y C") so the text matches the audio exactly. I'll add more Hindi, aiming for about 1 hour of clean audio.

**Recording setup:** MacBook mic (not Bluetooth), 48 kHz, short sessions with the same setup each time.

**Next (Day 2):** record session 1, write the benchmark sentences, and check that XTTS installs and trains on Kaggle.

## Day 2 (6 Oct): checking the setup, no recording

A quiet day. I checked the version pins from the night before on PyPI: coqui-tts 0.27.5 and transformers 4.57.6, with torch left to whatever Kaggle has. I also made sure my private planning files can't end up in the repo by mistake. No recording yet.

## Day 3 (7 Oct): Kaggle runner, recording tool, first T4 runs

**Kaggle runner.** One script pushes any repo script to a private Kaggle kernel at an exact commit, checks that the GPU is a T4, and downloads the output along with the commit it ran. Every number I report later can be traced back to the code that produced it.

**Environment check on a T4.** XTTS-v2 loads and speaks English and Hindi with a stock voice. Kaggle already had torch 2.11 with a matching torchcodec, so the pins are now final. The timings were one or two sentences in a stock voice, so they're a smoke test, not a result.

**Recording tool.** A small Mac script with a level meter, warnings for clipping and low level, redo/skip, and resume across sessions. A second script orders the prompts hardest-first and balances the languages, so even session 1 on its own is usable. It also flags anything likely to go over the 11.6 s clip limit in the XTTS training recipe (none did).

**Synthesis script.** It runs the benchmark through a model with fixed sampling settings and a fixed seed, and records speed and time to first audio. I've only run it on three throwaway sentences so far.

**Next:** record my reference clips and session 1, then rerun synthesis with my own voice as the reference.

## Day 4 (8 Oct): pipeline, session 1, first probe

**Pipeline.** Data prep (clip checks, trimming, loudness, Whisper transcript check, split), GPT fine-tuning and export, the text normaliser, and the evals (Whisper CER, ECAPA speaker similarity, duration checks). A training smoke test ran on a T4: 441 M of 518 M parameters trainable, resume works, the export loads. Three trainer problems turned up and were fixed: Kaggle's two GPUs, best-model saves filling the disk, and resume overwriting the step target.

**Benchmark frozen** the night before (88 sentences). I changed 24 prompts to masculine first person.

**Session 1:** 224 takes, about 16 minutes usable. Every take started with about 30 ms of digital zeros, which broke the noise-floor estimate; fixed. Whisper forced to English translates Hinglish, so its Hinglish CER is report-only. All 6 automatic drops were Whisper mishearing correct takes, so I keep them after listening.

**Probe:** the pretrained model reads Roman Hinglish badly but reads the same sentence well in Devanagari. That became the failure to fix.

## Day 5 (9 Oct): baselines and the first fine-tune

E0 (pretrained, raw text) crashes on 8 of 12 Hindi rows: XTTS's own digit expander fails on Hindi. E1 adds my normaliser and has no crashes, so E1 is the baseline. E2 (session 1, checkpoint picked by val CER) trained fine. A crashed row used to stop the whole run; now it's recorded and the run continues. Built the Hinglish-to-Devanagari step (IndicXlit plus a small lexicon).

## Day 6 (10 Oct): session 2, E2 fails, E3

**Session 2:** 113 takes, mostly Hindi and Hinglish. Final data: 333 clips, 24 minutes. I dropped the planned third session.

**E2 failed.** Val CER on 8 clips picked step 250 of 1000, and it sounded robotic and not like me. I changed the rule to lowest val loss with CER and duration guards. Also found: after fine-tuning, the model reads raw digits worse, so the normaliser stays on.

**E3** (both sessions, 2000 steps) picked step 1500 and sounded much better to me. Trained twice with the same seed: identical selection table. E2b (session 1, same recipe) picked step 1000.

**Listening test:** 22 blind pairs, done by three people together as one consensus session. E3 sounded more like me in 15 of 16 pairs, and Devanagari Hinglish won 4 of 4.

Lost about 85 GPU minutes to a file named `profile.py` that shadowed Python's own module.

## Day 7 (11 Oct): final numbers and write-up

Final comparison with 3 seeds and bootstrap intervals. Final (E3 + Hinglish route) vs pretrained: CER 0.125 → 0.061, almost all from Hinglish (0.400 → 0.098). Held-out Hinglish confirms the route. E4 (training on Devanagari Hinglish), a sampling sweep and long-sentence splitting gave nothing beyond noise, so none were adopted. E2b with 3 seeds: the extra 8 minutes only measurably helped Hinglish. Names are the main remaining weakness.

The Hinglish time-to-first-audio gap (0.57 s vs 0.43 s) is transliteration on Kaggle's CPU, which runs before synthesis. I added a word cache, so repeated text costs nothing; new words still cost the full time.

Wrote the report, experiment table and sample page.
