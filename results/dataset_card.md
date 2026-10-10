# Dataset card: desi-tts own-voice data

## Provenance

- **Speaker:** one adult male speaker (the author), recorded himself on a MacBook Air built-in mic, 48 kHz mono, in sessions of about 20 minutes. Consent: own voice.
- **Prompts p0001-p0480:** generated from templates by a script (`build_prompts.py`, written with AI assistance), then checked by script for spoken-form rules (no digits or symbols, acronyms as spaced letters).
- **Prompts p0481-p0650:** drafted by an AI assistant (8 Oct 2026) as a Hindi/Hinglish top-up, then checked by script for the same rules and for overlap with the benchmark.
- **Edits:** 24 prompts were changed by the AI assistant on the speaker's instruction to use masculine first-person forms, and two self-introductions with a name were made neutral (ids listed in the project notes). No other edits recorded as of 8 Oct 2026.
- **Sessions:** two recorded sessions (8 and 10 Oct 2026). A planned third session was dropped, so the set is about 24 minutes, not the 1 hour planned. Session 1 had 3 cut-off takes moved aside for re-recording; the recorder offered them again in session 2.
- **Hindi reading aid:** I read Hindi from an AI-written romanisation shown by the recorder; the saved transcript stays the Devanagari prompt. A misread romanisation would show up as a transcript mismatch in the Whisper check.
- **Whisper check and listening:** every automatic drop was listened to. All but one were Whisper mishearing a correct take, kept with the reason recorded in overrides.csv; the one real drop (p0293) was cut off.
- **Benchmark and reference sentences** are kept out of training; finalize.py checks this (leakage.csv).
- **Access:** private. The audio is never published; the Kaggle dataset is private.

## Size (clips used for training, after trimming)

| group | XTTS language | clips | hours | train | val |
|---|---|---|---|---|---|
| en | en | 154 | 0.20 | 146 | 8 |
| hinglish | en | 82 | 0.09 | 78 | 4 |
| hi | hi | 97 | 0.11 | 92 | 5 |
| **total** | | 333 | 0.40 | | |

## Quality

- Duration (s): median 4.2, min 1.6, max 8.4
- SNR (dB): median 37.2, min 25.2, max 44.1
- Status: {'keep': 288, 'flag': 45, 'drop': 1}

## Drops and flags by reason

| reason | dropped | flagged |
|---|---|---|
| listened: | 1 | 0 |
| long_pause | 0 | 5 |
| low_snr | 0 | 6 |
| transcript_borderline | 0 | 35 |
| transcript_mismatch | 1 | 0 |

Recorded masters (48 kHz) are kept unchanged; training copies are trimmed, loudness-normalised to -23 LUFS and resampled to 22.05 kHz mono.
