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
