# Озвучка через Google Translate TTS (gTTS) — запасной вариант, если edge-tts заблокирован.
# pip install gtts
import glob, os, re, time
from gtts import gTTS

MAX_CHARS = 180  # безопасный размер куска для одного запроса

def chunks(text):
    # режем по предложениям, склеиваем в куски <= MAX_CHARS
    parts = re.split(r'(?<=[.!?…])\s+', text)
    buf, out = '', []
    for p in parts:
        if len(buf) + len(p) + 1 > MAX_CHARS and buf:
            out.append(buf); buf = ''
        while len(p) > MAX_CHARS:  # очень длинное предложение — режем по словам
            cut = p.rfind(' ', 0, MAX_CHARS)
            if cut <= 0: cut = MAX_CHARS
            out.append(p[:cut]); p = p[cut:].strip()
        buf = (buf + ' ' + p).strip()
    if buf: out.append(buf)
    return out

for path in sorted(glob.glob("[0-9][0-9].txt")):
    out = path.replace(".txt", ".mp3")
    if os.path.exists(out) and os.path.getsize(out) > 1000:
        print("пропуск (уже есть):", out, flush=True)
        continue
    text = open(path, encoding="utf-8").read().strip()
    tmp = []
    ok = True
    for i, ch in enumerate(chunks(text)):
        f = f"_part{i}.mp3"
        try:
            gTTS(ch, lang='ru').save(f)
            tmp.append(f)
            time.sleep(0.5)
        except Exception as e:
            print(f"  ошибка в {out} часть {i}: {e}", flush=True)
            ok = False
            break
    if ok and tmp:
        with open(out, 'wb') as w:
            for f in tmp:
                w.write(open(f, 'rb').read())
        print("готово:", out, flush=True)
    for f in tmp:
        try: os.remove(f)
        except OSError: pass
    if not ok:
        try: os.remove(out)
        except OSError: pass
