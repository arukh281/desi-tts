# Days

Short log of what I did each day and why, including what I chose not to do.

## Day 1 (5 Oct): decisions, no training yet

**Compute: free Kaggle T4 (16 GB).** No budget, and the brief says more compute doesn't score higher. Because it's smaller than the 24–48 GB the brief assumes, I'll train only part of the model with a small batch, and save checkpoints often since Kaggle sessions can be cut.

**Model: XTTS-v2, fine-tuning only the GPT part.** It already speaks English and Hindi, clones a voice and can stream. The GPT part turns text into sound, so it carries the accent; the rest stays frozen to fit in 16 GB. It reads characters, not phonemes, so wrong pronunciations can often be fixed in the text. Not chosen: Indic Parler-TTS (already strong, harder to fine-tune on a T4) or training from scratch. Note: XTTS-v2's licence is non-commercial.

**Data: my own voice only.** None of the open sets I checked (IndicTTS, SYSPIN, Rasa, LIMMITS, Svarah) has one clean speaker covering Indian English, Hinglish and Hindi. Mixing in another speaker's Hindi would blur the voice. Hindi will have the least data, so I expect it to be the weak spot.

**Recording script: 480 sentences** (English, Hinglish, Hindi), mostly banking and support. Numbers and acronyms are written out as spoken ("K Y C") so the text matches the audio exactly. I'll add more Hindi, aiming for about 1 hour of clean audio.

**Recording setup:** MacBook mic (not Bluetooth), 48 kHz, short sessions with the same setup each time.

**Next (Day 2):** record session 1, write the benchmark sentences, and check that XTTS installs and trains on Kaggle.
