# Celal: Hermes'e bağlanan Türkçe sesli asistan (Mac)

Celal, Mac'te kendi penceresinde açılan sesli asistandır. Beyni sunucudaki
[Hermes](https://github.com/NousResearch/hermes-agent) ajanıdır; ses tarafı tamamen yereldir.
Yazarsın ya da konuşursun, Celal Hermes'e sorar, cevabı ekranda gösterir ve Türkçe sesle okur.

```
mikrofon → Silero VAD + Smart Turn v3 → Whisper large-v3-turbo MLX (tr)
        → SSH tüneli → Hermes API (/v1/responses, akış) → EMA Lightning (tr) / say -v Yelda → hoparlör
```

**Kulaklık gerekmez.** Celal konuşurken (ve son sesten sonraki ~400 ms boyunca) mikrofon
tamamen yok sayılır: VAD yok, STT yok, kesinti yok. Böylece hoparlörden çıkan kendi sesini
duyup susmaz. Konuşurken araya girmek istersen `--barge-in` (kulaklıkla kullan).

**Mikrofon varsayılan olarak kapalıdır.** Pencere açılınca Celal dinlemez; yalnızca
**Dinle** düğmesine bastığında ya da **ctrl+space** kısayolunda dinler (aynı tuş kapatır).
Dinlerken düğme kırmızıdır, kırmızı nokta yanıp söner, menü çubuğunda `⏺` görünür.
Odadaki video/TV sesi de konuşma sayılır; işin bitince mikrofonu kapat.

## Kurulum

Gereken: Apple Silicon Mac, Python 3.12 (uv kurar), [uv](https://docs.astral.sh/uv/),
sunucuya SSH erişimi, Hermes API anahtarı, model diski.

```bash
git clone <repo> celal && cd celal
uv python install 3.12        # yoksa
uv sync
uv run pytest                 # mock'larla, model indirmez, ağa çıkmaz
```

### 1. Model diski

Bütün model ağırlıkları ve önbellekler (Whisper MLX, EMA Lightning, Hugging Face hub, torch)
`CELAL_MODELS_DIR` altında durur; varsayılan `/Volumes/KIOXIA/Modeller`. Celal `HF_HOME`,
`HF_HUB_CACHE`, `XDG_CACHE_HOME`, `TORCH_HOME` değişkenlerini oraya yönlendirir.
Klasör yoksa Celal açılmaz ve şunu yazar:

```
hata: model diski takılı değil: /Volumes/KIOXIA/Modeller bulunamadı. ...
```

İç diske sessizce bir şey indirilmez. (Mikrofonsuz `--text --tts say` model istemez, disk olmadan da
çalışır; `--mic` Whisper'a ihtiyaç duyduğu için diski ister.)
İlk çalıştırmada eksik modeller diske iner: Whisper turbo ~1,6 GB, EMA Lightning ~34 MB.
Smart Turn v3 ve Silero Pipecat paketinin içinde gelir.

### 2. Sunucu bağlantısı

`~/.config/celal/config.toml` (repo dışında, makineye özel):

```toml
[ssh]
host = "kullanici@sunucu-adresi"
key = "~/.ssh/sunucu_anahtari"

[hermes]
local_port = 8642        # Mac'teki uç
remote_port = 8642       # sunucuda Hermes API (yalnız 127.0.0.1)
conversation = "celal-mac"
api = "responses"        # sohbet sunucuda tutulur; "chat" = /v1/chat/completions, geçmiş Mac'te

[ui]
hotkey = "ctrl+space"    # mikrofon aç/kapat; "off" kapatır
```

API anahtarı sunucuda `~/.config/celal/api-key` dosyasındadır. Mac'e bir kez kopyala,
ekrana yazdırma:

```bash
mkdir -p ~/.config/celal && chmod 700 ~/.config/celal
scp -i ~/.ssh/sunucu_anahtari kullanici@sunucu-adresi:.config/celal/api-key ~/.config/celal/api-key
chmod 600 ~/.config/celal/api-key
```

Celal SSH tünelini (`ssh -N -L 8642:127.0.0.1:8642 …`) kendisi arka planda açar, koparsa
yeniden kurar, istekler takılırsa tüneli öldürüp yeniden başlatır. Port zaten açıksa
(elle açılmış tünel) onu kullanır. Anahtar yalnız `Authorization` başlığına girer; loglanmaz.

Ortam değişkenleri dosyayı ezer: `CELAL_SSH_HOST`, `CELAL_SSH_KEY`, `CELAL_PORT`,
`CELAL_REMOTE_PORT`, `CELAL_CONVERSATION`, `CELAL_API`, `CELAL_API_KEY_FILE`,
`CELAL_HOTKEY`, `CELAL_MODELS_DIR`, `CELAL_CONFIG`.

### 3. İzinler

macOS ilk açılışta mikrofon izni sorar (izin, komutu çalıştıran uygulamaya verilir: Terminal,
iTerm…). Kısayol (Carbon hotkey) ek izin istemez ve menü çubuğu sürecinde yaşar
(`--no-menubar` ile kısayol da olmaz). Başka bir uygulama aynı tuşu kaydetmişse pencerede
uyarı çıkar. macOS'un kendi kısayolları (ör. ctrl+space = "önceki giriş kaynağı") bu
kontrolde görünmez: tuşa basınca mikrofon değişmiyorsa `[ui] hotkey` ile başka bir tuş seç
(ör. `ctrl+option+space`).

## Çalıştırma

```bash
uv run celal                      # pencere + menü çubuğu simgesi (mikrofon kapalı açılır)
uv run celal --mic                # pencere, mikrofon açık başlar
uv run celal --text               # terminal: yaz, Enter, Celal cevaplar ve konuşur
echo "merhaba Celal" | uv run celal --text --out /tmp/celal.wav   # sesi WAV'a da yaz
uv run celal --text --mic         # terminal + mikrofon (hoparlör testi)
uv run celal --bench              # Whisper / TTS / RAM ölçümü
```

Pencere:

- Üstte sunucu durumu (**bağlı** / **bağlı değil**, sebebi ve tünel durumu) ve Celal'in hâli:
  **hazır / dinliyor / düşünüyor / konuşuyor**. Hermes ilk kelimeye kadar 5–15 s düşünebilir;
  cevap akarken cümle cümle okunur.
- **Dinle**: mikrofonu açar/kapatır (ctrl+space da aynı işi yapar).
- **■ Durdur** (ya da Esc): konuşmayı keser ve sunucudaki isteği iptal eder.
- **Yeni sohbet**: sunucuda yeni bir sohbet adı açar.
- Sunucu erişilemezse pencere "bağlı değil" gösterir, mesaj gönderilmez, uygulama çökmez;
  tünel dönünce kendiliğinden "bağlı"ya geçer.

Menü çubuğu (`● Celal` bağlı, `○ Celal` bağlı değil, `⏺` dinliyor): pencereyi göster/gizle,
mikrofon aç/kapat, Durdur, **Çıkış**. Çıkış tüneli kapatır ve uygulamayı temiz kapatır.
Pencerenin kapat düğmesi pencereyi gizler (menüden geri açılır). Terminalde Ctrl-C da
(pencere ve `--text` modunda) tüneli kapatıp çıkar.

Bayraklar:

| Bayrak | Anlamı |
|---|---|
| `--text` | terminal modu (stdin'den yazı) |
| `--mic` | mikrofon açık başlasın |
| `--barge-in` | Celal konuşurken araya girince sussun (kulaklık gerekir) |
| `--min-words N` | `--barge-in` ile: N kelimeden sonra sus (0: sesini duyunca) |
| `--tts ema\|say` | TTS motoru (varsayılan `ema`) |
| `--stt turbo\|turbo-q4` | Whisper modeli |
| `--out dosya.wav` | çalınan sesi WAV'a da yaz |
| `--no-play` | hoparlöre çalma |
| `--no-menubar` | menü çubuğu simgesi olmasın (kısayol da kapanır) |
| `--timeout N` | `--text`: cevap için en çok N saniye (varsayılan 300) |
| `--log debug\|info\|warning` | log seviyesi |

## Test ederken

- Gerçek sunucuyla denerken ayrı bir sohbet adı kullan, Mirza'nın geçmişi kirlenmesin:
  `CELAL_CONVERSATION=celal-test-$(date +%s) uv run celal --text`
- Bitince test oturumunu sunucuda sil: `hermes sessions delete --yes <oturum-id>`
  (id: `hermes sessions list --source api_server`), saklanan cevabı da API'den sil:
  `DELETE /v1/responses/<resp_id>`.
- Kendi başlattığın her süreci kapat (Celal, tünel); mikrofonu açık bırakma.
  Menüden **Çıkış** ya da Ctrl-C tüneli de kapatır; `pgrep -fl "ssh -N -L 8642"` boş dönmeli.

## Nasıl çalışır

- `src/celal/tunnel.py`: `TunnelManager`, ssh alt sürecini izler, koparsa geri açar (1-2-5-10 s).
- `src/celal/hermes.py`: `HermesClient`, `/v1/responses` (`conversation` ile sohbet sunucuda)
  ya da `/v1/chat/completions`, ikisi de SSE akışı. Sesli kullanım için kısa cevap yönergesi gönderir.
- `src/celal/llm_hermes.py`: Hermes'i Pipecat LLM servisi yapar; TTS'e giden metinden
  markdown ve emoji temizlenir (`speech_text.py`), ekranda asıl metin görünür.
- `src/celal/echo_gate.py`: konuşma penceresi boyunca mikrofon sesini ve transkriptleri düşürür.
- `src/celal/core.py`: tünel + sağlık kontrolü + pipeline; pencere ve terminal aynı çekirdeği kullanır.
- `src/celal/ui.py` + `web/index.html`: pywebview penceresi; `menubar.py`: rumps simgesi
  (ayrı süreç, çünkü iki kütüphane de Cocoa döngüsünü ister); `hotkey.py`: Carbon kısayolu.
- Faz 1'den kalanlar: `audio_io.py` (sounddevice transport), `tts_ema.py`, `tts_say.py`,
  `stt_filter.py` (Whisper'ın "Altyazı M.K." uydurmaları), `timing.py`, `bench.py`.

## Bilinen sınırlar

- Barge-in kapalıyken Celal konuşurken söylediğin duyulmaz; durdurmak için **Durdur**/Esc.
- Whisper parça bazlıdır; konuşman bitince ~1–2 s'de transkript gelir.
- Hermes ilk kelimeyi 5–15 s'de verir (sunucu tarafı ajan + araçlar).
- `responses` modunda Durdur akışı keser; sunucu o turu yine de sohbete yazabilir.
- Celal SIGKILL ile ölürse ssh tüneli açık kalabilir; bir sonraki açılış onu "harici tünel" olarak kullanır.

## Lisans

MIT
