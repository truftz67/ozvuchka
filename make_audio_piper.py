# Озвучка мужским голосом через Piper (локально, без интернета после скачивания модели).
# Голос: ru_RU-dmitri-medium (мужской). 01.mp3 и 02.mp3 уже сделаны edge-tts — не трогаем.
import glob, os, wave, io
import numpy as np
from piper import PiperVoice
from lameenc import Encoder

VOICE_PATH = "voices/ru_RU-dmitri-medium.onnx"
RATE = 0.95  # чуть медленнее для доклада

def synth(text):
    voice = PiperVoice.load(VOICE_PATH)
    pcm = bytearray()
    for chunk in voice.synthesize(text):
        pcm += chunk.audio_int16_bytes
    return bytes(pcm), voice.config.sample_rate

def to_mp3(pcm, rate):
    enc = Encoder()
    enc.set_bit_rate(128)
    enc.set_in_sample_rate(rate)
    enc.set_channels(1)
    enc.set_quality(2)
    return enc.encode(pcm) + enc.flush()

for path in sorted(glob.glob("[0-9][0-9].txt")):
    num = os.path.basename(path)[:2]
    if num in ("01", "02"):   # уже озвучены Дмитрием через edge-tts
        print("пропуск:", path, flush=True)
        continue
    out = path.replace(".txt", ".mp3")
    text = open(path, encoding="utf-8").read().strip()
    pcm, rate = synth(text)
    mp3 = to_mp3(pcm, rate or 22050)
    with open(out, 'wb') as f:
        f.write(mp3)
    print("готово:", out, flush=True)
