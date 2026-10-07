"""Make a tiny training set from a stock XTTS voice, for pipeline smoke tests only.

About 40 short clips (en + hi) in the same layout data_prep/finalize.py writes:
wavs/<id>.wav at 22.05 kHz, metadata_{en,hi}_{train,val}.csv, plus clips.csv
for data_prep/asr_check.py. One extra clip, smoke_wrong, has a transcript that
doesn't match its audio, so the Whisper check has something to catch. It's
left out of the training metadata. Never used for a real result.

Usage (on Kaggle):  python train/make_smoke_data.py --out DIR [--speaker "Ana Florence"]
"""
import argparse
import csv
import os
from pathlib import Path

EN = """Thanks for calling, how can I help you today?
Your order has been packed and will ship tomorrow.
Please hold while I check the details.
The technician will reach your home in the evening.
Your new card has been dispatched.
We have noted your complaint and will call you back.
The branch is closed on public holidays.
Your refund will reflect in three working days.
Could you please confirm your registered email?
Your plan includes unlimited calls and messages.
I am sorry for the trouble this has caused.
The delivery was rescheduled at your request.
Your request has been forwarded to our team.
Please keep your account details ready.
The app update fixes the login problem.
Thank you for your patience and have a nice day.
We could not reach you on your phone earlier.
Your appointment has been moved to next week.
The payment link will expire in one hour.
Is there anything else I can help you with?""".splitlines()
HI = """नमस्ते, मैं आपकी क्या मदद कर सकता हूँ?
आपका ऑर्डर कल तक पहुँच जाएगा।
कृपया थोड़ी देर लाइन पर बने रहिए।
आपकी शिकायत हमने दर्ज कर ली है।
आपका नया कार्ड भेज दिया गया है।
हमारी टीम आपको जल्दी ही कॉल करेगी।
आपका रिफंड तीन दिन में आ जाएगा।
क्या आप अपना ईमेल दोबारा बता सकते हैं?
आज शाम को तकनीशियन आपके घर आएँगे।
आपकी अपॉइंटमेंट अगले हफ़्ते के लिए बदल दी गई है।
भुगतान का लिंक एक घंटे में खत्म हो जाएगा।
आपका धन्यवाद, आपका दिन शुभ हो।
हम आपसे पहले संपर्क नहीं कर पाए।
आपकी जानकारी सुरक्षित रखी जाएगी।
कृपया अपना खाता नंबर तैयार रखिए।
ऐप के नए वर्ज़न में यह समस्या ठीक हो गई है।
आपका प्लान अगले महीने रिन्यू होगा।
क्या मैं आपकी और कोई मदद कर सकता हूँ?
आपकी डिलीवरी आपके कहने पर बदल दी गई है।
हमें इस परेशानी के लिए खेद है।""".splitlines()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(os.environ.get("OUT_DIR", "outputs"), "smoke_data"))
    ap.add_argument("--speaker", default="Ana Florence")
    args = ap.parse_args()

    os.environ["COQUI_TOS_AGREED"] = "1"
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly
    from TTS.api import TTS

    out = Path(args.out)
    (out / "wavs").mkdir(parents=True, exist_ok=True)
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
    rows = [(f"smoke_en_{i:02d}", "en", t) for i, t in enumerate(EN)]
    rows += [(f"smoke_hi_{i:02d}", "hi", t) for i, t in enumerate(HI)]
    torch.manual_seed(0)
    for clip_id, lang, text in rows + [("smoke_wrong", "en", EN[0])]:
        spoken = EN[1] if clip_id == "smoke_wrong" else text  # audio says a different sentence
        wav = tts.tts(text=spoken, speaker=args.speaker, language=lang)
        sf.write(out / "wavs" / f"{clip_id}.wav", resample_poly(wav, 147, 160), 22050)  # 24 kHz -> 22.05 kHz

    with open(out / "clips.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "lang", "text"])
        for clip_id, lang, text in rows + [("smoke_wrong", "en", EN[0])]:
            w.writerow([clip_id, lang, text])
    for lang in ("en", "hi"):
        mine = [r for r in rows if r[1] == lang]
        for part, subset in (("train", mine[2:]), ("val", mine[:2])):
            with open(out / f"metadata_{lang}_{part}.csv", "w", encoding="utf-8") as f:
                f.write("audio_file|text|speaker_name\n")
                f.writelines(f"wavs/{i}.wav|{t}|smoke\n" for i, _, t in subset)
    print(f"wrote {len(rows) + 1} clips to {out}")


if __name__ == "__main__":
    main()
