# Озвучка слайдов бесплатными голосами Microsoft (edge-tts).
# Запуск: 1) pip install edge-tts   2) положите этот файл рядом с 01.txt ... 15.txt   3) python make_audio.py
# Голос: ru-RU-DmitryNeural (мужской) или ru-RU-SvetlanaNeural (женский)
import asyncio, glob, os, time
import edge_tts

VOICE = "ru-RU-DmitryNeural"
RATE = "-5%"   # чуть медленнее для доклада; "+0%" — обычный темп

async def synth(text, out, tries=6):
    for i in range(tries):
        try:
            await edge_tts.Communicate(text, VOICE, rate=RATE).save(out)
            return True
        except Exception as e:
            print(f"  попытка {i+1} не удалась: {e}")
            await asyncio.sleep(5 * (i + 1))
    return False

async def main():
    for path in sorted(glob.glob("[0-9][0-9].txt")):
        out = path.replace(".txt", ".mp3")
        if os.path.exists(out) and os.path.getsize(out) > 1000:
            print("пропуск (уже есть):", out)
            continue
        text = open(path, encoding="utf-8").read().strip()
        ok = await synth(text, out)
        print(("готово: " if ok else "ОШИБКА: ") + out, flush=True)
        await asyncio.sleep(3)

asyncio.run(main())