# Озвучечка — веб-сервис озвучки текста (Flask + Piper TTS, много языков).
# Функции: txt/pdf/ручной ввод, 39 голосов, авторизация, история, отзывы, авто-перевод.
# Запуск: python app.py  ->  http://127.0.0.1:5000
import io, json, os, sqlite3, threading, time, urllib.parse, urllib.request
import datetime
from collections import OrderedDict
from pathlib import Path

from flask import (Flask, jsonify, redirect, render_template, request,
                   send_file, session, url_for)
from piper import PiperVoice
from piper.config import SynthesisConfig
from lameenc import Encoder
from werkzeug.security import check_password_hash, generate_password_hash

BASE = Path(__file__).parent
VOICES_DIR = BASE / "voices"
DATA_DIR = BASE / "data"
AUDIO_DIR = DATA_DIR / "audio"
DB_PATH = DATA_DIR / "app.db"
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
app.secret_key = os.environ.get("SECRET_KEY", "ozvuchka-secret-change-me")
_lock = threading.Lock()
_cache = OrderedDict()

# ---------- БД ----------
def db():
    DATA_DIR.mkdir(exist_ok=True)
    AUDIO_DIR.mkdir(exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("""CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL,
        pass_hash TEXT NOT NULL, created REAL NOT NULL)""")
    con.execute("""CREATE TABLE IF NOT EXISTS history(
        id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL,
        text TEXT NOT NULL, voice TEXT NOT NULL, audio BLOB,
        created REAL NOT NULL)""")
    con.execute("""CREATE TABLE IF NOT EXISTS reviews(
        id INTEGER PRIMARY KEY, username TEXT NOT NULL, rating INTEGER NOT NULL,
        text TEXT NOT NULL, created REAL NOT NULL)""")
    # миграция: audio было именем файла -> теперь BLOB
    cols = {r[1]: r[2] for r in con.execute("PRAGMA table_info(history)")}
    if cols.get("audio") == "TEXT":
        con.execute("ALTER TABLE history RENAME TO history_old")
        con.execute("""CREATE TABLE history(
            id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL,
            text TEXT NOT NULL, voice TEXT NOT NULL, audio BLOB,
            created REAL NOT NULL)""")
        for row in con.execute("SELECT * FROM history_old"):
            blob = None
            if row[4]:
                f = AUDIO_DIR / row[4]
                if f.exists():
                    blob = f.read_bytes()
            con.execute("INSERT INTO history VALUES(?,?,?,?,?,?)",
                        (row[0], row[1], row[2], row[3], blob, row[5]))
        con.execute("DROP TABLE history_old")
    return con

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
def split_text(text, limit):
    import re
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

def detect_lang(text):
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0
        code = detect(text[:3000])
    except Exception:
        return None
    # langdetect путает кириллицу: если есть чисто русские буквы — это русский
    import re
    if code in {"uk", "bg", "sr", "mk", "be", "sh", "cnr", "hr", "sl", "bs", "mk"} \
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

def current_user():
    uid = session.get("uid")
    if not uid:
        return None
    con = db()
    row = con.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
    con.close()
    return row

# ---------- Страницы ----------
@app.get("/")
def index():
    con = db()
    def fmt(rows):
        out = []
        for r in rows:
            d = dict(r)
            d["created"] = datetime.datetime.fromtimestamp(d["created"]).strftime("%d.%m.%Y %H:%M")
            out.append(d)
        return out
    reviews = fmt(con.execute("SELECT username, rating, text, created FROM reviews ORDER BY id DESC LIMIT 20").fetchall())
    history = []
    user = current_user()
    if user:
        history = fmt(con.execute(
            "SELECT id, text, voice, audio, created FROM history WHERE user_id=? ORDER BY id DESC LIMIT 50",
            (user["id"],)).fetchall())
    con.close()
    return render_template("index.html",
                           voices={k: v["label"] for k, v in VOICES.items()},
                           user=user, reviews=reviews, history=history)

@app.post("/register")
def register():
    username = (request.form.get("username") or "").strip()[:32]
    password = request.form.get("password") or ""
    if len(username) < 2 or len(password) < 4:
        return "Логин мин. 2 символа, пароль мин. 4", 400
    con = db()
    try:
        con.execute("INSERT INTO users(username, pass_hash, created) VALUES(?,?,?)",
                    (username, generate_password_hash(password), time.time()))
        con.commit()
    except sqlite3.IntegrityError:
        con.close()
        return "Такой логин уже занят", 400
    uid = con.execute("SELECT id FROM users WHERE username=?", (username,)).fetchone()["id"]
    con.close()
    session["uid"] = uid
    return redirect(url_for("index"))

@app.post("/login")
def login():
    username = (request.form.get("username") or "").strip()
    password = request.form.get("password") or ""
    con = db()
    row = con.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    con.close()
    if row and check_password_hash(row["pass_hash"], password):
        session["uid"] = row["id"]
        return redirect(url_for("index"))
    return "Неверный логин или пароль", 400

@app.post("/logout")
def logout():
    session.pop("uid", None)
    return redirect(url_for("index"))

@app.post("/review")
def review():
    username = (request.form.get("username") or "Гость").strip()[:32] or "Гость"
    try:
        rating = max(1, min(5, int(request.form.get("rating", 5))))
    except ValueError:
        rating = 5
    text = (request.form.get("text") or "").strip()[:1000]
    if text:
        con = db()
        con.execute("INSERT INTO reviews(username, rating, text, created) VALUES(?,?,?,?)",
                    (username, rating, text, time.time()))
        con.commit()
        con.close()
    return redirect(url_for("index") + "#reviews")

# ---------- API ----------
def do_generate(text, voice, tempo, do_translate, filename=None):
    if not text:
        return None, "Пустой текст"
    if len(text) > MAX_TEXT:
        return None, f"Слишком длинный текст (макс. {MAX_TEXT} символов)"
    if voice not in VOICES:
        return None, "Неизвестный голос"
    final_text = text
    if do_translate:
        try:
            final_text = translate_text(text, VOICES[voice]["lang"])
        except Exception as e:
            return None, f"Ошибка перевода: {e}"
    try:
        with _lock:
            mp3 = synth_mp3(final_text, voice, tempo)
    except Exception as e:
        return None, str(e)
    return (mp3, final_text), None

def save_history(user_id, text, voice, mp3):
    con = db()
    con.execute("INSERT INTO history(user_id, text, voice, audio, created) VALUES(?,?,?,?,?)",
                (user_id, text[:2000], voice, mp3, time.time()))
    con.commit()
    con.close()

@app.post("/api/generate")
def generate():
    body = request.get_json(force=True)
    text = (body.get("text") or "").strip()
    voice = body.get("voice", DEFAULT_VOICE)
    tempo = body.get("tempo", "normal")
    do_translate = bool(body.get("translate", True))
    result, err = do_generate(text, voice, tempo, do_translate)
    if err:
        return jsonify(error=err), 500 if "перевода" in err else 400
    mp3, _ = result
    user = current_user()
    if user:
        try:
            save_history(user["id"], text, voice, mp3)
        except Exception:
            pass
    return send_file(io.BytesIO(mp3), mimetype="audio/mpeg", download_name="voice.mp3")

@app.post("/api/generate_file")
def generate_file():
    voice = request.form.get("voice", DEFAULT_VOICE)
    tempo = request.form.get("tempo", "normal")
    do_translate = request.form.get("translate", "1") == "1"
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify(error="Файл не выбран"), 400
    try:
        text = extract_text(f)
    except Exception as e:
        return jsonify(error=f"Не удалось прочитать файл: {e}"), 400
    result, err = do_generate(text, voice, tempo, do_translate)
    if err:
        return jsonify(error=err), 500 if "перевода" in err else 400
    mp3, _ = result
    user = current_user()
    if user:
        try:
            save_history(user["id"], text, voice, mp3)
        except Exception:
            pass
    return send_file(io.BytesIO(mp3), mimetype="audio/mpeg",
                     download_name=Path(f.filename).stem + ".mp3")

@app.get("/audio/<int:hid>")
def audio(hid):
    user = current_user()
    if not user:
        return "not found", 404
    con = db()
    row = con.execute("SELECT audio FROM history WHERE id=? AND user_id=?",
                      (hid, user["id"])).fetchone()
    con.close()
    if not row or row["audio"] is None:
        return "not found", 404
    return send_file(io.BytesIO(row["audio"]), mimetype="audio/mpeg")

if __name__ == "__main__":
    VOICES_DIR.mkdir(exist_ok=True)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
