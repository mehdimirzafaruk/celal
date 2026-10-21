# Vera (jarvis): yerel Türkçe sesli asistan, Faz 1

Mikrofondan Türkçe dinler, Claude ile (Claude Code aboneliği, Agent SDK) konuşur,
Türkçe sesle cevap verir. Sen araya girince susar. Kulaklık varsayılır; echo iptali yok.

```
mikrofon → Silero VAD + Smart Turn v3 (lokal) → Whisper large-v3-turbo MLX (tr)
        → Claude Agent SDK (haiku, araçsız) → EMA Lightning (tr) / say -v Yelda → hoparlör
```

## Kurulum

Gereken: Apple Silicon Mac, [uv](https://docs.astral.sh/uv/), giriş yapılmış Claude Code CLI
(`claude` → `/login`). API anahtarı gerekmez.

```bash
cd ~/Kod/jarvis
uv python install 3.12   # yoksa
uv sync
uv run pytest            # mock'larla, model indirmez
```

İlk çalıştırmada modeller `~/.cache/huggingface` altına iner: Whisper turbo ~3 GB,
turbo-q4 ~0,9 GB, EMA Lightning ~66 MB. Smart Turn v3 ve Silero Pipecat içinde gelir.

macOS ilk `run`'da mikrofon izni sorabilir (izin, komutu çalıştıran uygulamaya verilir:
Terminal, iTerm, Claude...). İzin yoksa ses hep sıfır gelir; program bunu çıkışta yazar.

`ANTHROPIC_API_KEY` ortamda tanımlıysa CLI onu kullanabilir. Aboneliği garanti etmek için:
`env -u ANTHROPIC_API_KEY uv run jarvis …`

## Komutlar

```bash
uv run jarvis run                         # mikrofon → Vera → hoparlör (Ctrl-C ile çık)
uv run jarvis run --tts say               # EMA yerine macOS Yelda sesi
uv run jarvis run --stt turbo-q4          # daha az RAM'li Whisper
uv run jarvis run --model sonnet          # haiku yerine sonnet
uv run jarvis run --log debug             # Pipecat ayrıntılı log

echo "Merhaba Vera, bugün ne yapıyoruz?" | uv run jarvis text --out /tmp/vera.wav
uv run jarvis text                        # etkileşimli: yaz, Enter, Vera konuşsun

uv run jarvis bench                       # Whisper doğruluk/süre, TTS TTFA/RTF, tepe RAM
```

Bayraklar:

| Bayrak | Komut | Anlamı |
|---|---|---|
| `--tts ema\|say` | run, text | TTS motoru (varsayılan `ema`) |
| `--stt turbo\|turbo-q4` | run | Whisper modeli (varsayılan `turbo`) |
| `--model haiku\|sonnet\|<id>` | run, text | Claude modeli (varsayılan `haiku`) |
| `--min-words N` | run | Vera konuşurken kesmek için en az kelime (varsayılan 2) |
| `--out dosya.wav` | run, text | Çalınan sesi WAV'a da yaz |
| `--no-play` | run, text | Hoparlöre çalma (akış gerçek zamanlı sürer) |
| `--wav dosya.wav` | run | Mikrofon yerine WAV çal (test) |
| `--duration N` | run | N saniye sonra kapan (test) |
| `--log debug\|info\|warning` | hepsi | Log seviyesi |

Her turda konsolda:

```
[sen] Merhaba Vera bugün ne yapıyoruz?
[vera] Merhaba Mirza!
⏱ stt=1.47s first_token=2.16s ttfa=0.18s total=3.81s
```

`stt`: konuşmanın bitişi → transkript. `first_token`: transkript → Claude'un ilk kelimesi.
`ttfa`: ilk kelime → ilk ses. `total`: konuşmanın bitişi → ilk ses. `text` modunda `stt=-`.

## Nasıl çalışır

- `src/jarvis/llm_claude.py`: `ClaudeAgentLLMService`. Tek `ClaudeSDKClient` oturum boyunca
  açık kalır (çok turlu bağlam Claude tarafında). Araçsız (`tools=[]`), ayar izolasyonu
  (`setting_sources=[]`, `strict_mcp_config=True`, `skills=[]`), `include_partial_messages=True`.
  `text_delta` → `LLMTextFrame`, `ResultMessage` → `LLMFullResponseEndFrame`. Kesintide
  `client.interrupt()` çağrılır, kesilen tur `ResultMessage` gelene kadar boşaltılır.
- `src/jarvis/tts_ema.py`: EMA Lightning `stream()` (CPU) bir thread'de; float32 parçalar int16
  `TTSAudioRawFrame` olur. `src/jarvis/tts_say.py`: `say -v Yelda` cümle başına WAV üretir.
- `src/jarvis/pipeline.py`: Pipecat 1.13 pipeline'ı; kesinti kuralı
  `MinWordsUserTurnStartStrategy(min_words=2)`, tur sonu `LocalSmartTurnAnalyzerV3`.
- `src/jarvis/audio_io.py`: `sounddevice` tabanlı yerel ses transport'u (aşağıya bak).
- `src/jarvis/stt_filter.py`: Whisper'ın sessizlikte uydurduğu "Altyazı M.K." gibi altyazı
  kalıplarını siler.

## Bilinen sınırlar

- **Kulaklık şart.** Echo iptali yok; hoparlörden çıkan ses mikrofona girerse Vera kendini keser.
- **Kesinti, sen durunca tetiklenir.** Whisper parça bazlı (ara transkript yok). "En az 2 kelime"
  kuralı ancak cümlen bitip transkript gelince sayılır; Vera o ana kadar konuşmaya devam eder
  (cümle süresi + ~1,3 s). Gerçek zamanlı kesinti için Faz 2'de streaming STT gerekir.
- **Whisper'ın sabit maliyeti ~1,2 s.** Encoder her parçayı 30 s pencereye doldurur; "Evet."
  bile ~1,2 s sürer. turbo-q4 hızı değiştirmez, yalnız RAM/disk azaltır.
- **Kısa parçalar düşebilir.** Virgülde duraklayınca VAD cümleyi böler; "Tamam dur," gibi
  1–2 kelimelik parça Whisper'dan boş dönebilir.
- Sayılar rakamla yazılır ("9'da"); TTS bunları okur ama WER'e hata olarak girer.
- Pipecat'in `LocalAudioTransport`'u PyAudio ister, PyAudio da `brew install portaudio` ister.
  Sistem paketi kurmamak için aynı yapıda `sounddevice` transport'u kullanılıyor (PortAudio
  wheel'in içinde gelir).
- `say` akış yapmaz; ilk ses tüm cümle üretilince gelir (~1 s).
