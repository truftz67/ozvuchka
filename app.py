# Веб-озвучка текста: txt/pdf файлы или ручной ввод, голоса Piper (много языков).
# Запуск: python app.py  ->  http://127.0.0.1:5000
import io, json, os, threading, urllib.request
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file
from piper import PiperVoice
from piper.config import SynthesisConfig
from lameenc import Encoder

BASE = Path(__file__).parent
VOICES_DIR = BASE / "voices"
HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main/"

# Отобранные голоса: русский + другие национальности/языки
SELECT = [
    # Русский
    "ru_RU-dmitri-medium", "ru_RU-denis-medium", "ru_RU-ruslan-medium", "ru_RU-irina-medium",
    # Украинский / казахский / болгарский / сербский (кириллица — читают и русский с акцентом)
    "uk_UA-ukrainian_tts-medium", "uk_UA-mykyta-high", "uk_UA-tetiana-high",
    "kk_KZ-issai-high", "kk_KZ-iseke-x_low", "kk_KZ-raya-x_low",
    "bg_BG-dimitar-medium", "sr_RS-serbski_institut-medium",
    # Английский
    "en_US-lessac-medium", "en_US-libritts_r-medium", "en_US-kathleen-low",
    "en_GB-cori-medium", "en_GB-southern_english_female-low",
    # Немецкий
    "de_DE-thorsten-medium", "de_DE-thorsten_emotional-medium", "de_DE-kerstin-medium",
    # Французский / испанский / итальянский / португальский
    "fr_FR-siwis-medium", "fr_FR-tom-medium",
    "es_ES-sharvard-medium", "es_ES-davefx-medium",
    "it_IT-serena-medium", "it_IT-riccardo-x_low",
    "pt_BR-faber-medium", "pt_BR-cadu-medium",
    # Польский / турецкий / японский / корейский / китайский
    "pl_PL-gosia-medium", "pl_PL-darkman-medium",
    "tr_TR-dfki-medium",
    "ja_JP-hi_fi_captain-medium",
    "ko_KR-kss-medium",
    "zh_CN-chaowen-medium", "zh_CN-xiao_ya-medium",
    # Хинди / арабский / шведский
    "hi_IN-rohan-medium", "hi_IN-priyamvada-medium",
    "ar_JO-kareem-medium",
    "sv_SE-alma-medium",
]

GENDER_RU = {"dmitri": "мужской", "denis": "мужской", "ruslan": "мужской", "irina": "женский"}

_meta = json.loads((BASE / "voices.json").read_text(encoding="utf-8"))
VOICES = {}
for key in SELECT:
    e = _meta[key]
    onnx_path = next(p for p in e["files"] if p.endswith(".onnx"))
    lang = e["language"]["name_english"]
    name = e["name"]
    if e["language"]["code"].startswith("ru") and name in GENDER_RU:
        label = f"Русский — {name.capitalize()} ({GENDER_RU[name]})"
    else:
        label = f"{lang} — {name}"
    VOICES[key] = {"label": label, "onnx": onnx_path}

DEFAULT_VOICE = "ru_RU-dmitri-medium"
TEMPO = {"slow": 1.15, "normal": 1.0, "fast": 0.85}  # length_scale

app = Flask(__name__)
_lock = threading.Lock()
_cache = {}

def download_voice(info):
    for suffix in ("", ".json"):
        src = info["onnx"] + suffix
        dst = VOICES_DIR / Path(src).name
        if not dst.exists():
            print(f"Скачиваю голос ({src})...", flush=True)
            urllib.request.urlretrieve(HF_BASE + src, dst)

def get_voice(key):
    if key not in _cache:
        info = VOICES[key]
        onnx = VOICES_DIR / Path(info["onnx"]).name
        if not onnx.exists():
            download_voice(info)
        _cache[key] = PiperVoice.load(str(onnx))
    return _cache[key]

def synth_mp3(text, voice_key, tempo="normal"):
    voice = get_voice(voice_key)
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

def check_voice(key):
    if key not in VOICES:
        raise ValueError("Неизвестный голос")

@app.get("/")
def index():
    return render_template("index.html", voices={k: v["label"] for k, v in VOICES.items()})

@app.post("/api/generate")
def generate():
    body = request.get_json(force=True)
    text = (body.get("text") or "").strip()
    voice = body.get("voice", DEFAULT_VOICE)
    tempo = body.get("tempo", "normal")
    if not text:
        return jsonify(error="Пустой текст"), 400
    if len(text) > 20000:
        return jsonify(error="Слишком длинный текст (макс. 20000 символов)"), 400
    try:
        check_voice(voice)
        with _lock:
            mp3 = synth_mp3(text, voice, tempo)
    except Exception as e:
        return jsonify(error=str(e)), 500
    return send_file(io.BytesIO(mp3), mimetype="audio/mpeg", download_name="voice.mp3")

@app.post("/api/generate_file")
def generate_file():
    voice = request.form.get("voice", DEFAULT_VOICE)
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
        check_voice(voice)
        with _lock:
            mp3 = synth_mp3(text, voice, tempo)
    except Exception as e:
        return jsonify(error=str(e)), 500
    return send_file(io.BytesIO(mp3), mimetype="audio/mpeg", download_name=Path(f.filename).stem + ".mp3")

if __name__ == "__main__":
    VOICES_DIR.mkdir(exist_ok=True)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
