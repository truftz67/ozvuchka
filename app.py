# Веб-озвучка текста: txt/pdf файлы или ручной ввод, голоса Piper (ru_RU).
# Запуск: python app.py  ->  http://127.0.0.1:5000
import io, os, threading, urllib.request
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file
from piper import PiperVoice
from piper.config import SynthesisConfig
from lameenc import Encoder

BASE = Path(__file__).parent
VOICES_DIR = BASE / "voices"

VOICES = {
    "dmitri": "Дмитрий (мужской)",
    "denis": "Денис (мужской)",
    "ruslan": "Руслан (мужской)",
    "irina": "Ирина (женский)",
}
TEMPO = {"slow": 1.15, "normal": 1.0, "fast": 0.85}  # length_scale

app = Flask(__name__)
HF = "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/{v}/medium/ru_RU-{v}-medium.onnx{ext}"

def download_voice(name):
    for ext in ("", ".json"):
        url = HF.format(v=name, ext=ext)
        dst = VOICES_DIR / f"ru_RU-{name}-medium.onnx{ext}"
        if not dst.exists():
            print(f"Скачиваю голос {name}...", flush=True)
            urllib.request.urlretrieve(url, dst)

_lock = threading.Lock()
_cache = {}

def get_voice(name):
    if name not in _cache:
        onnx = VOICES_DIR / f"ru_RU-{name}-medium.onnx"
        if not onnx.exists():
            download_voice(name)
        _cache[name] = PiperVoice.load(str(onnx))
    return _cache[name]

def synth_mp3(text, voice_name, tempo="normal"):
    voice = get_voice(voice_name)
    cfg = SynthesisConfig(length_scale=TEMPO.get(tempo, 1.0))
    pcm = bytearray()
    rate = voice.config.sample_rate
    for chunk in voice.synthesize(text, syn_config=cfg):
        pcm += chunk.audio_int16_bytes
    enc = Encoder()
    enc.set_bit_rate(128)
    enc.set_in_sample_rate(rate)
    enc.set_channels(1)
    enc.set_quality(2)
    return enc.encode(bytes(pcm)) + enc.flush()

def pdf_text(data):
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    return "\n".join((p.extract_text() or "") for p in reader.pages).strip()

@app.get("/")
def index():
    return render_template("index.html", voices=VOICES)

@app.post("/api/generate")
def generate():
    body = request.get_json(force=True)
    text = (body.get("text") or "").strip()
    voice = body.get("voice", "dmitri")
    tempo = body.get("tempo", "normal")
    if not text:
        return jsonify(error="Пустой текст"), 400
    if len(text) > 20000:
        return jsonify(error="Слишком длинный текст (макс. 20000 символов)"), 400
    try:
        with _lock:
            mp3 = synth_mp3(text, voice, tempo)
    except Exception as e:
        return jsonify(error=str(e)), 500
    return send_file(io.BytesIO(mp3), mimetype="audio/mpeg", download_name="voice.mp3")

@app.post("/api/generate_file")
def generate_file():
    voice = request.form.get("voice", "dmitri")
    tempo = request.form.get("tempo", "normal")
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify(error="Файл не выбран"), 400
    data = f.read()
    name = f.filename.lower()
    try:
        if name.endswith(".pdf"):
            text = pdf_text(data)
        else:
            text = data.decode("utf-8-sig", errors="replace").strip()
    except Exception as e:
        return jsonify(error=f"Не удалось прочитать файл: {e}"), 400
    if not text:
        return jsonify(error="В файле не найден текст"), 400
    if len(text) > 20000:
        return jsonify(error="Слишком длинный текст (макс. 20000 символов)"), 400
    try:
        with _lock:
            mp3 = synth_mp3(text, voice, tempo)
    except Exception as e:
        return jsonify(error=str(e)), 500
    return send_file(io.BytesIO(mp3), mimetype="audio/mpeg", download_name=Path(f.filename).stem + ".mp3")

if __name__ == "__main__":
    VOICES_DIR.mkdir(exist_ok=True)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
