# desi-tts

Fine-tuning an open TTS model for Indian English, with Hindi / Hinglish support.

This is still a work in progress, so things will move around a bit. I'll keep updating this README as I go.

## What I'm trying to do

Take an existing pretrained TTS model and make it better at the things Indian voice apps actually need:

- Indian English accent and natural prosody
- Indian names and places
- numbers, dates, currency (₹, lakh/crore) and acronyms like KYC, EMI, UPI
- Hinglish, e.g. "Aapka payment successfully receive ho gaya hai"

## Plan

- [ ] Collect and clean ~2-3 hours of speech data
- [ ] Pick a base model and generate baseline audio on a fixed benchmark set
- [ ] Fine-tune on the cleaned data
- [ ] Run a few experiments (data filtering, checkpoint selection, targeted data)
- [ ] Compare baseline vs fine-tuned: listening tests, pronunciation, intelligibility, speaker consistency
- [ ] Find one clear failure and fix it
- [ ] Measure inference speed and write up the report

## Repo layout

Will add this once the code is in.

## Notes

Datasets and model weights aren't stored in this repo. I'll add download scripts and setup instructions here.
