# Озвучечка — веб-сервис озвучки текста (Flask + Piper TTS, много языков).
# Функции: txt/pdf/ручной ввод, 39 голосов, авто-перевод на язык голоса.
# Запуск: python app.py  ->  http://127.0.0.1:5000
import io, json, os, re, threading, time, urllib.parse, urllib.request
from collections import OrderedDict
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file
from piper import PiperVoice
from piper.config import SynthesisConfig
from lameenc import Encoder

BASE = Path(__file__).parent
VOICES_DIR = BASE / "voices"
HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
MM_URL = "https://api.mymemory.translated.net/get?q={text}&langpair={src}|{dst}"
LINGVA_URLS = [
    "https://lingva.ml/api/v1/{src}/{dst}/{text}",
    "https://lingva.lunar.icu/api/v1/{src}/{dst}/{text}",
    "https://translate.plausibility.cloud/api/v1/{src}/{dst}/{text}",
]
MAX_TEXT = 100_000   # поддержка больших файлов
SYN_CHUNK = 900      # размер куска для синтеза
TR_CHUNK = 1000      # размер куска для перевода
_tr_cache = {}

# Отобранные голоса
SELECT = [
    "ru_RU-dmitri-medium", "ru_RU-denis-medium", "ru_RU-ruslan-medium", "ru_RU-irina-medium",
    "uk_UA-ukrainian_tts-medium", "uk_UA-mykyta-high", "uk_UA-tetiana-high",
    "kk_KZ-issai-high", "kk_KZ-iseke-x_low", "kk_KZ-raya-x_low",
    "bg_BG-dimitar-medium", "sr_RS-serbski_institut-medium",
    "en_US-lessac-medium", "en_US-libritts_r-medium", "en_US-kathleen-low",
    "en_GB-cori-medium", "en_GB-southern_english_female-low",
    "de_DE-thorsten-medium", "de_DE-thorsten_emotional-medium", "de_DE-kerstin-medium",
    "fr_FR-siwis-medium", "fr_FR-tom-medium",
    "es_ES-sharvard-medium", "es_ES-davefx-medium",
    "it_IT-serena-medium", "it_IT-riccardo-x_low",
    "pt_BR-faber-medium", "pt_BR-cadu-medium",
    "pl_PL-gosia-medium", "pl_PL-darkman-medium",
    "tr_TR-dfki-medium",
    "ja_JP-hi_fi_captain-medium",
    "ko_KR-kss-medium",
    "zh_CN-chaowen-medium", "zh_CN-xiao_ya-medium",
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
    VOICES[key] = {
        "label": label,
        "onnx": onnx_path,
        "lang": e["language"]["code"].split("_")[0],
    }

DEFAULT_VOICE = "ru_RU-dmitri-medium"
TEMPO = {"slow": 1.15, "normal": 1.0, "fast": 0.85}
MAX_CACHE = 2  # не больше 2 моделей в памяти — на Render free всего 512 МБ ОЗУ

app = Flask(__name__)
_lock = threading.Lock()
_cache = OrderedDict()

# ---------- Голоса ----------
def download_voice(info):
    for suffix in ("", ".json"):
        src = info["onnx"] + suffix
        dst = VOICES_DIR / Path(src).name
        if not dst.exists():
            print(f"Скачиваю голос ({src})...", flush=True)
            urllib.request.urlretrieve(HF_BASE + src, dst)

def get_voice(key):
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]
    info = VOICES[key]
    onnx = VOICES_DIR / Path(info["onnx"]).name
    if not onnx.exists():
        download_voice(info)
    voice = PiperVoice.load(str(onnx))
    _cache[key] = voice
    while len(_cache) > MAX_CACHE:
        _cache.popitem(last=False)
    return voice

# ---------- Текст ----------
def split_text(text, limit):
    parts = re.split(r"(?<=[.!?…\n])\s+", text)
    out, buf = [], ""
    for p in parts:
        while len(p) > limit:
            cut = p.rfind(" ", 0, limit)
            if cut <= 0:
                cut = limit
            out.append(p[:cut]); p = p[cut:].strip()
        if len(buf) + len(p) + 1 > limit and buf:
            out.append(buf); buf = ""
        buf = (buf + " " + p).strip()
    if buf:
        out.append(buf)
    return out

def synth_mp3(text, voice_key, tempo="normal"):
    voice = get_voice(voice_key)
    cfg = SynthesisConfig(length_scale=TEMPO.get(tempo, 1.0))
    rate = voice.config.sample_rate
    enc = Encoder()
    enc.set_bit_rate(128)
    enc.set_in_sample_rate(rate)
    enc.set_channels(1)
    enc.set_quality(2)
    out = bytearray()
    for part in split_text(text, SYN_CHUNK):
        pcm = bytearray()
        for chunk in voice.synthesize(part, syn_config=cfg):
            pcm += chunk.audio_int16_bytes
        out += enc.encode(bytes(pcm))
    out += enc.flush()
    return bytes(out)

# ---------- Перевод ----------
def detect_lang(text):
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0
        code = detect(text[:3000])
    except Exception:
        return None
    # langdetect путает кириллицу: если есть чисто русские буквы — это русский
    if code in {"uk", "bg", "sr", "mk", "be", "sh", "cnr", "hr", "sl", "bs"} \
            and re.search(r"[ыэъёЫЭЪЁ]", text):
        return "ru"
    return code

def _http_json(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "ozvuchka/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))

def _mm_chunk(chunk, src, dst):
    url = MM_URL.format(text=urllib.parse.quote(chunk), src=src, dst=dst)
    data = _http_json(url)
    if data.get("quotaFinished") or data.get("responseStatus") != 200:
        raise RuntimeError("лимит бесплатного переводчика исчерпан")
    return data["responseData"]["translatedText"]

def _lingva_chunk(chunk, src, dst):
    for tpl in LINGVA_URLS:
        try:
            url = tpl.format(src=src, dst=dst, text=urllib.parse.quote(chunk))
            data = _http_json(url, timeout=20)
            if isinstance(data, list) and data and data[0].get("translation"):
                return "".join(t["translation"] for t in data)
        except Exception:
            continue
    raise RuntimeError("lingva недоступен")

def _translate_chunk(chunk, src, dst):
    """MyMemory с ретраями на 429, затем запасные Lingva-инстансы, кэш."""
    key = (src, dst, chunk)
    if key in _tr_cache:
        return _tr_cache[key]
    last_err = None
    for attempt in range(4):
        try:
            res = _mm_chunk(chunk, src, dst)
            _tr_cache[key] = res
            return res
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code == 429:
                time.sleep(3 * (attempt + 1))
                continue
            break
        except Exception as e:
            last_err = e
            break
    try:
        res = _lingva_chunk(chunk, src, dst)
        _tr_cache[key] = res
        return res
    except Exception as e:
        raise RuntimeError(f"переводчик недоступен: {last_err or e}")

def translate_text(text, target_lang):
    """Переводит текст на язык голоса через MyMemory (авто-определение языка)."""
    target = {"zh": "zh-CN", "pt": "pt-BR"}.get(target_lang, target_lang)
    src = detect_lang(text)
    if not src:
        raise RuntimeError("не удалось определить язык текста")
    src = {"zh-cn": "zh-CN", "zh-tw": "zh-TW"}.get(src, src)
    if src == target or src.split("-")[0] == target.split("-")[0]:
        return text  # уже на нужном языке
    out = []
    for c in split_text(text, TR_CHUNK):
        out.append(_translate_chunk(c, src, target))
        time.sleep(0.4)
    return " ".join(out).strip()

# ---------- Файлы ----------
def pdf_text(data):
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    return "\n".join((p.extract_text() or "") for p in reader.pages).strip()

def extract_text(file_storage):
    data = file_storage.read()
    if file_storage.filename.lower().endswith(".pdf"):
        return pdf_text(data)
    return data.decode("utf-8-sig", errors="replace").strip()

# ---------- Страницы / API ----------
@app.get("/")
def index():
    return render_template("index.html",
                           voices={k: v["label"] for k, v in VOICES.items()})

def do_generate(text, voice, tempo):
    if not text:
        return None, "Пустой текст"
    if len(text) > MAX_TEXT:
        return None, f"Слишком длинный текст (макс. {MAX_TEXT} символов)"
    if voice not in VOICES:
        return None, "Неизвестный голос"
    final_text = text
    try:
        final_text = translate_text(text, VOICES[voice]["lang"])
    except Exception as e:
        # переводчик недоступен — озвучиваем исходный текст без ошибки
        print(f"Перевод не удался, использую оригинал: {e}", flush=True)
    try:
        with _lock:
            mp3 = synth_mp3(final_text, voice, tempo)
    except Exception as e:
        return None, str(e)
    return mp3, None

@app.post("/api/generate")
def generate():
    body = request.get_json(force=True)
    text = (body.get("text") or "").strip()
    voice = body.get("voice", DEFAULT_VOICE)
    tempo = body.get("tempo", "normal")
    mp3, err = do_generate(text, voice, tempo)
    if err:
        return jsonify(error=err), 400
    return send_file(io.BytesIO(mp3), mimetype="audio/mpeg", download_name="voice.mp3")

@app.post("/api/generate_file")
def generate_file():
    voice = request.form.get("voice", DEFAULT_VOICE)
    tempo = request.form.get("tempo", "normal")
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify(error="Файл не выбран"), 400
    try:
        text = extract_text(f)
    except Exception as e:
        return jsonify(error=f"Не удалось прочитать файл: {e}"), 400
    mp3, err = do_generate(text, voice, tempo)
    if err:
        return jsonify(error=err), 400
    return send_file(io.BytesIO(mp3), mimetype="audio/mpeg",
                     download_name=Path(f.filename).stem + ".mp3")

@app.get("/ads.txt")
def ads_txt():
    # замените pub-XXX на ваш publisher ID из AdSense
    return "google.com, pub-XXXXXXXXXXXXXXXX, DIRECT, f08c47fec0942fa0\n", 200, {"Content-Type": "text/plain"}

if __name__ == "__main__":
    VOICES_DIR.mkdir(exist_ok=True)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
