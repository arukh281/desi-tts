# Days

A running log of what I did each day, how, and why. I'm also writing down what I chose not to do, because those decisions shaped the project as much as the things I did.

---

## Day 1 (5 Oct): planning, compute, model and data decisions

No training or recording today. The whole day went on deciding things, so that the later days are mostly execution.

### What I did

**1. Read the brief and broke it into steps.**
Data → model → baseline → training → 3+ experiments → evaluation → one failure fixed → production numbers. The brief cares more about reasoning than about final voice quality, so I want each step to have a clear reason I can explain.

**2. Picked where to train: free Kaggle GPUs.**
- Kaggle gives a T4 (16 GB) for about 30 GPU-hours a week, with a 12-hour limit per session.
- *Why:* I didn't want to spend money on this, and the brief says more compute doesn't earn a higher score.
- *What it means for the next steps:*
  - The T4 is smaller than the 24–48 GB the brief assumes, so I'll use a small batch size and train only part of the model.
  - Sessions can get killed, so I'll save checkpoints regularly and test resuming early on.
- *Not done:* renting a bigger GPU. Faster, but not needed for this scope.

**3. Picked a model: XTTS-v2, fine-tuning only its GPT part.**
- XTTS-v2 already speaks English and Hindi, can copy a voice from a short clip, and can stream audio. The streaming matters for the latency part of the brief.
- It works in three steps:
  1. A GPT model reads the text and predicts audio tokens.
  2. The tokens are a compressed code for short slices of sound.
  3. A vocoder (HiFi-GAN) turns them into a waveform.
- It reads text as characters (BPE), not phonemes. So if it mispronounces something, the cheapest fix is often to change the text that goes in.
- I'll train only the GPT part, because that's the part that decides how text turns into sound, so it carries the accent. The rest stays frozen, which keeps memory use within 16 GB.
- *Not chosen:*
  - **Training a model from scratch.** It isn't possible on this compute, and the brief says not to.
  - **Indic Parler-TTS.** It's already good at Indian English, so there's less room to show improvement, and it's harder to fine-tune on a 16 GB T4.
- *Caveat:* the XTTS-v2 licence is non-commercial. That's fine for this assignment, but a real product would need a different model.
- *Not tested yet.* Next I'll check that it installs and fine-tunes on Kaggle at all, before I commit to it.

**4. Decided on data: my own voice only.**
I looked at the open datasets first:

| Dataset | Why I didn't use it |
|---|---|
| IndicTTS Hindi (IIT Madras) | Good studio data, but a different speaker from me, and the licence is research-only |
| SYSPIN, AI4Bharat Rasa | Clean studio Hindi, but no English, and again a different voice |
| LIMMITS | Has Hindi and English, but from different speakers, and needs a sign-up |
| Svarah | Indian English, but only about 5 minutes per speaker and not studio quality |
| Hinglish sets (MUCS etc.) | Lecture or ASR audio, not clean enough for TTS |

- No open dataset had **one clean speaker** covering Indian English, Hinglish and Hindi. That combination is what I need, and Hinglish in particular barely exists as clean TTS data.
- Mixing in someone else's Hindi would teach the model two voices. Its Hindi could start drifting towards that other speaker.
- So I'll record my own voice: one speaker, and data I own with no licence questions.
- *Trade-off:* Hindi will have the least data, so I expect it to be the weak spot. I'll measure that rather than hide it.
- *Kept for later:* adding about an hour of IndicTTS Hindi could be an experiment ("does borrowed Hindi help pronunciation but hurt voice consistency?"). Only if time allows.

**5. Prepared the recording script: 480 sentences.**
- The mix is about 300 Indian English, 120 Hinglish (in Latin script) and 60 Hindi (in Devanagari).
- The content is mostly banking and support (EMI, KYC, UPI, loans, appointments) plus everyday conversation. That's close to what a voice agent actually says.
- Numbers, amounts and acronyms are written out the way I say them ("twelve thousand rupees", "K Y C"). The model learns from characters, so the text has to match the audio exactly. A "₹" in the text with "rupees" in the audio would teach it the wrong thing.
- I kept sentences short enough to stay under about 11 seconds, the limit XTTS trains on.
- I'll add about 170 more sentences, mostly Hindi, because 60 Hindi lines is thin. The target is about 1 hour of clean audio after filtering.

**6. Decided how to record.**
- **Mic:** the MacBook mic in a quiet room, not my Bluetooth earbuds. Bluetooth mics drop to phone-call quality while recording.
- **Sample rate:** record at 48 kHz and downsample later. I can lower quality afterwards, but I can't add it back.
- **Sessions:** two or three of 20–25 minutes each, with the same room, distance and speaking style every time. A tired voice in a long session would get learned too.
- **Room tone:** 5 seconds of silence first, so I know the room's noise level. I'll need it for the noise filter later.

### What I didn't do today
- No recording. I was in the office, so it moves to tomorrow.
- No code run on Kaggle yet.

### How this shapes the next steps
- **Day 2:**
  - Record the first session (about 20 minutes).
  - Write the benchmark sentences myself, and keep them separate from the training sentences so the test stays fair.
  - Run a small test on Kaggle: install XTTS, generate a few clips with the original model, then run a few training steps to check memory and speed.
- After that: clean the recordings, generate the baseline, then train.
- The rule I'm going by: **clean data over more data**. Every clip I drop will be logged with a reason, and one experiment will check whether dropping those clips actually helped.
