# Celal: Mac'te çalışan Türkçe sesli asistan

Celal, Mac'te kendi penceresinde açılan sesli asistandır. Yazarsın ya da konuşursun; mesaj
**önce Mac'te**, Mac'teki Claude Code'da çalışır (kendi skill'lerin, hook'ların, MCP'lerin,
CLAUDE.md'n ile). Yalnız gerçekten sunucuda yapılması gereken iş sunucudaki
[Hermes](https://github.com/NousResearch/hermes-agent) ajanına gider. Ses tarafı tamamen yereldir.

```
mikrofon → Silero VAD + Smart Turn v3 → Whisper large-v3-turbo MLX (tr) ┐
yazı ────────────────────────────────────────────────────────────────────┤
                                                                         ▼
                 aktif sekme → Mac'teki Claude Code (Agent SDK, sekme başına kalıcı oturum)
                                 └─ sunucu işi gerekirse: celal hermes "<istek>" → Hermes
                 "sunucuda …" → Hermes API (SSH tüneli, /v1/responses, akış)
                                                                         ▼
                               EMA Lightning (tr) / say -v Yelda → hoparlör (yalnız aktif sekme)
Hermes (Telegram) mac_task → ssh -R → Celal'in yerel API'si (127.0.0.1:8643) → yeni 📱 sekme
```

**Kulaklık gerekmez.** Celal konuşurken (ve son sesten sonraki ~400 ms boyunca) mikrofon
tamamen yok sayılır: VAD yok, STT yok, kesinti yok. Böylece hoparlörden çıkan kendi sesini
duyup susmaz. Konuşurken araya girmek istersen `--barge-in` (kulaklıkla kullan).

**Mikrofon varsayılan olarak kapalıdır.** Pencere açılınca Celal dinlemez; yalnızca
**Dinle** düğmesine bastığında ya da **ctrl+space** kısayolunda dinler (aynı tuş kapatır).
Dinlerken düğme kırmızıdır, kırmızı nokta yanıp söner, menü çubuğunda `⏺` görünür.
Odadaki video/TV sesi de konuşma sayılır; işin bitince mikrofonu kapat.

**Açık mikrofon "Hey Siri" gibi bekler.** Mikrofon açılınca Celal pasiftir (durum `bekliyor`,
düğme sarı, menü çubuğunda `◌`): Whisper yerelde çalışır ama duyduğu hiçbir şey Hermes'e ya da
Mac'e gitmez. Yalnız şu iki cümle tam olarak söylenince durum değişir:

- **"Celal dinlemeye başla"** → aktif dinleme (`dinliyor`, `⏺`); artık her söz Celal'e gider.
- **"Celal dinlemeyi durdur"** → yeniden pasif. İki cümle de Hermes'e mesaj olarak gitmez.

Büyük/küçük harf, baştaki/sondaki noktalama ("Celal, dinlemeye başla.") ve boşluk fark etmez;
fazladan ya da eksik kelime, benzer cümleler ("Celal dinlemeye başlasana") tetiklemez.
Mikrofonu düğmeyle kapatıp açınca yine pasif başlar. Eski davranış (açık mikrofon = hep dinle)
için `--no-wake`.

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

[mac]                    # varsayılan beyin: Mac'teki Claude Code (hepsi isteğe bağlı, değerler varsayılan)
model = "sonnet"         # hız için; "opus" da yazılabilir
cwd = "~"                # yeni sekmelerin çalışma klasörü
claude = "claude"        # ya da tam yol; PATH'e ek olarak ~/.local/bin aranır
timeout = 1800           # bir cevap için en çok saniye
idle_close = 600         # boşta kalan Claude süreci kapanır, oturum saklanır
max_parallel = 3         # aynı anda en çok kaç sekme çalışır (pencere + Hermes ortak)
api = true               # Hermes → Mac yerel API'si ve ters tünel
api_port = 8643
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
`CELAL_HOTKEY`, `CELAL_MODELS_DIR`, `CELAL_CONFIG`, `CELAL_HERMES_PREFIXES` (virgüllü),
`CELAL_MAC_MODEL`, `CELAL_MAC_CWD`, `CELAL_CLAUDE`, `CELAL_MAC_TIMEOUT`, `CELAL_MAC_IDLE_CLOSE`,
`CELAL_MAC_MAX_PARALLEL`, `CELAL_MAC_WARM` (0/1), `CELAL_MAC_API` (0/1), `CELAL_MAC_PORT`,
`CELAL_MAC_API_KEY_FILE`, `CELAL_STATE_DIR` (`tabs.json`, `tasks.json`; varsayılan `~/.config/celal`).

### 2b. Mac'teki Claude Code ve Hermes → Mac

Mac'te giriş yapılmış Claude Code gerekir (`claude` komutu; Max aboneliğiyle giriş). Celal onu
[Claude Agent SDK](https://pypi.org/project/claude-agent-sdk/) ile sekme başına bir `ClaudeSDKClient`
olarak açar: `permission_mode="bypassPermissions"`, `include_partial_messages=True`, `setting_sources`
verilmez (yani `~/.claude` altındaki ayar, skill, hook, MCP ve CLAUDE.md'lerin hepsi yüklenir), sistem
istemine Türkçe kısa sesli cevap kuralı ve "sunucu işi için `celal hermes`" kuralı eklenir. Celal açılırken
kendi ortamındaki `CLAUDE*` / `ANTHROPIC_*` değişkenlerini siler (`CLAUDE_CODE_OAUTH_TOKEN` hariç):
alt süreç başka bir Claude oturumunun içinde sanmasın, API faturası değil Max girişi kullanılsın.

Hermes'in de Mac'e iş verebilmesi için (Hermes → Mac):

1. İlk açılışta Celal `~/.config/celal/mac-api-key` dosyasını üretir (mode 600). Sunucuya bir kez
   kopyala, ekrana yazdırma:
   ```bash
   ssh -i ~/.ssh/sunucu_anahtari kullanici@sunucu-adresi 'umask 077; mkdir -p ~/.config/celal; cat > ~/.config/celal/mac-api-key' < ~/.config/celal/mac-api-key
   ```
2. Sunucudaki Hermes'te `celal-tools` eklentisi `http://127.0.0.1:8643` adresini bu anahtarla çağırır
   (aşağıda "Hermes → Mac API").
3. Celal ayrı bir `ssh -N -R 127.0.0.1:8643:127.0.0.1:8643` süreci açar. SSH anahtarı
   `authorized_keys`'te `permitlisten` ile kısıtlıysa listeye `permitlisten="127.0.0.1:8643"` eklenmeli.

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

- Üstte Hermes bağlantısı (**Hermes bağlı** / **bağlı değil**) ve Celal'in hâli:
  **hazır / bekliyor / dinliyor / düşünüyor / konuşuyor**. Hermes bağlı değilken de Mac işleri çalışır.
- **Sekme şeridi**: her sekme bir görev/konuşma (bir Claude oturumu). Etiket görevin ilk kelimeleri;
  durum işareti: dönen halka çalışıyor, `…` sırada, ✓ bitti, `!` hata, ■ durduruldu; 📱 Hermes'ten geldi;
  mavi nokta okunmamış sonuç. **+ Yeni** yeni sekme açar; sağ tık ya da × kapatır (çalışıyorsa önce durdurur).
- **■ Durdur** (ya da Esc): aktif sekmedeki Mac görevini keser ve sesi susturur. Düğme hemen
  "durduruluyor…" olur, sekmede "durduruluyor" yazar, bitince "Durdur: Mac görevi durduruldu, ses kesildi."
  notu çıkar. Sunucuya gönderilmiş işlere (Hermes sekmeleri, `celal hermes`, Telegram'dan başlayan sunucu
  görevleri) iptal göndermez; onlarda yalnız ses kesilir. 📱 sekmeler Mac'te koştuğu için Durdur onları durdurur.
- Alt satırda `2/3 çalışıyor · 1 sırada` gibi paralel iş sayısı.

Menü çubuğu (`● Celal` bağlı, `○ Celal` bağlı değil, `⏺` dinliyor): pencereyi göster/gizle,
mikrofon aç/kapat, Durdur (aktif sekme), **Çıkış**. Çıkış tüneli kapatır ve uygulamayı temiz kapatır.
Pencerenin kapat düğmesi pencereyi gizler (menüden geri açılır). Terminalde Ctrl-C da
(pencere ve `--text` modunda) tüneli kapatıp çıkar.

Bayraklar:

| Bayrak | Anlamı |
|---|---|
| `--text` | terminal modu (stdin'den yazı; her satır aktif sekmeye) |
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

## Yönlendirme: önce Mac

| Mesaj | Nereye |
|---|---|
| ön eksiz (yazılı ya da sesli) | aktif sekmenin Mac'teki Claude Code'u; aktif sekme Hermes sekmesiyse en son boştaki Mac sekmesi ya da yeni sekme |
| `sunucuda …`, `Hermes'e sor …` | doğrudan Hermes (Hermes sekmesi açılır ya da kullanılır) |
| `yeni görev …` | yeni sekme |
| eski `Mac'te …` ön eki | Mac (ön ek atılır) |

Mac'teki Claude sunucu işini kendisi devreder: Hermes cron'ları, Yasomi içerik hattı, sunucudaki servisler,
Telegram'a mesaj gibi işlerde `celal hermes "<istek>"` çalıştırır (sunucuya SSH gerekiyorsa hermes-server
skill'ini kullanır). Mac'te yapılabilecek iş sunucuya gönderilmez.

```bash
echo "bu klasörde kaç dosya var" | CELAL_MAC_CWD=~/Kod/celal uv run celal --text --tts say --no-play
```

```
[sen] bu klasörde kaç dosya var
[mac] sekme “bu klasörde kaç dosya” (/Users/…/Kod/celal)
[mac] Claude hazır (claude-sonnet-…)…
[mac] komut çalıştırıyor: find . -maxdepth 1 -type f | wc -l…
[celal] Bu klasörde 54 dosya var; .git ve .venv saymadım.
⏱ ilk kelime 3.6 s, ilk ses 4.7 s, tam cevap 3.6 s
```

Ölçüm (24 Ekim 2026, M2, `say` TTS; EMA ilk sesi ~1 s daha erken verir): Mac'te "merhaba" ilk ses 2,6 s,
"bu klasörde kaç dosya var" 4,7 s, aynı sekmede takip sorusu 3,2 s (süreç yeniden açılmaz). Aynı makinede
Hermes yolu ("sunucuda merhaba") 10,5 s; eski Hermes-varsayılan sürüm 7–15 s.

## Sekmeler ve paralel görevler

- Her sekmenin kendi Claude oturumu (`ClaudeSDKClient`) ve kuyruğu vardır. Sekme meşgulken yazılan mesaj
  o sekmenin sırasına girer ("sırada"), cevap bitince aynı oturuma gider.
- Aynı anda en çok `max_parallel` (3) sekme çalışır; fazlası "sırada" bekler. Sınır pencere ve Hermes'ten
  gelen işler için ortaktır.
- Yalnız **aktif sekmenin** cevabı sesli okunur, cümle cümle, akarken (en çok ~700 karakter, sonra
  "Devamı ekranda."). Arka plandaki sekme bitince tek cümle okunur ("README görevi bitti") ve sekmeye
  okunmamış noktası gelir. Hermes'ten gelen 📱 sekmeler bitince ses çıkarmaz (sonucu Hermes iletir).
  İlerleme satırları ("dosya okuyor…", "komut çalıştırıyor: pytest") sekmede görünür, okunmaz.
- Claude bir komutu arka plana atarsa (ör. uzun `sleep`), Celal o iş bitip Claude raporlayana kadar sekmeyi
  "çalışıyor" tutar; Durdur arka plandaki işi de öldürür (Claude süreci kapanır, oturum saklanır).
- 10 dk boşta kalan Claude süreci kapanır; `session_id` saklanır, sonraki mesaj `resume` ile aynı sohbete döner.
- Sekmeler kalıcıdır: `~/.config/celal/tabs.json` (etiket, session_id, cwd, son durum, son 50 satır).
  Celal açılınca geri gelir; açılışta aktif sekmenin Claude'u önceden başlatılır (ilk mesaj daha hızlı).

## Hermes → Mac API

127.0.0.1:8643, her istekte `Authorization: Bearer <mac-api-key>`. Hiçbir uç beklemez.

```
POST /task              {"task", "cwd"?, "source"?, "label"?}
                        → 202 {"task_id", "tab_id", "status": "running"|"queued", "queued": bool, "cwd", "label"}
GET  /task/<id>         → {"status": queued|running|done|error|cancelled, "result", "error",
                           "started_at", "finished_at", "progress": [...], ...}
POST /task/<id>/cancel  → {"cancelled": bool}
GET  /tasks             → {"tasks": [çalışan / sıradaki sekmeler], "running", "queued", "max_parallel"}
GET  /health            → {"ok": true, "busy", "running", "queued", "max_parallel"}
```

Her görev pencerede yeni bir 📱 sekmede, pencereden yazılmış gibi Mac'teki Claude'da koşar (odak
değişmez). Sonuçlar `~/.config/celal/tasks.json`'da 24 saat tutulur; Celal yeniden başlasa da
`GET /task/<id>` döner (yarıda kalan görev `error` olur).

## `celal hermes`

```bash
uv run celal hermes "sunucudaki hermes cron listesini ver"
```

Hermes API'sine (model `celal`, sohbet `<conversation>-claude`) tek istek atar, cevabı akarken stdout'a
yazar; hata stderr'e, çıkış kodu 1/2. Tünel açık değilse geçici bir `ssh -L` açar, iş bitince kapatır.
Model diski gerekmez. Mac'teki Claude sunucu işini bununla devreder; Hermes'e "işi Mac'e geri gönderme"
denir.

**Güvenlik:** Mac'teki Claude izin sormadan çalışır (`bypassPermissions`): dosya silebilir, komut
çalıştırabilir. Bu yola yalnız iki giriş var: Celal penceresi/terminali ve Hermes. Hermes'e yalnız
Mirza'nın Telegram'ı ve Celal ulaşır (API sunucuda yalnız 127.0.0.1'de); yerel API yalnız 127.0.0.1'de
dinler ve her istekte `mac-api-key` bearer anahtarını ister (sabit zamanlı karşılaştırma). Anahtar hiçbir
yere loglanmaz. Bu zinciri genişletme (Hermes'e başka kanal açma, API'yi dış ağa açma) Mac'i de açar.

## Test ederken

- Gerçek sunucuyla denerken ayrı bir sohbet adı ve ayrı sekme deposu kullan, Mirza'nın geçmişi ve
  sekmeleri kirlenmesin:
  `CELAL_CONVERSATION=celal-test-$(date +%s) CELAL_STATE_DIR=/tmp/celal-test uv run celal --text`
- Bitince test oturumunu sunucuda sil: `hermes sessions delete --yes <oturum-id>`
  (id: `hermes sessions list --source api_server`), saklanan cevabı da API'den sil:
  `DELETE /v1/responses/<resp_id>`.
- Kendi başlattığın her süreci kapat (Celal, tünel); mikrofonu açık bırakma.
  Menüden **Çıkış** ya da Ctrl-C tüneli de kapatır; `pgrep -fl "ssh -N -L 8642"` ve
  `pgrep -fl "ssh -N -R 127.0.0.1:8643"` boş dönmeli.
- Hermes → Mac denemesi (sunucuda): `curl -s -X POST -H "Authorization: Bearer $(cat ~/.config/celal/mac-api-key)"
  -d '{"task":"uname -a çalıştır"}' http://127.0.0.1:8643/task` hemen `task_id` döner, pencerede 📱 sekme açılır;
  `GET /task/<id>` sonucu verir. Celal kapalıyken bağlantı reddedilir.

## Nasıl çalışır

- `src/celal/core.py`: tünel + sağlık kontrolü + sekmeler + yönlendirme + ses; pencere ve terminal aynı
  çekirdeği kullanır. Yalnız aktif sekmenin cevabını cümle cümle `TTSSpeakFrame` ile okur.
- `src/celal/tabs.py`: `TabManager` (sekme başına kuyruk ve işçi, ortak `max_parallel` semaforu, Durdur,
  boşta kapanma, `tabs.json`) ve `TaskStore` (`tasks.json`, 24 saat).
- `src/celal/brains.py`: `MacClaudeBrain` (Agent SDK `ClaudeSDKClient`; okuyucu görev, arka plan işlerini
  bekleme, kesme) ve `HermesBrain` (tek Hermes isteği; Durdur iptal etmez).
- `src/celal/mac_task.py`: yönlendirme kuralı (`route`), ön ek ve klasör ayrıştırma, Türkçe adım satırları.
- `src/celal/mac_api.py`: 127.0.0.1:8643'teki küçük HTTP API (bearer anahtar).
- `src/celal/pipeline.py`: Pipecat yalnız kulak ve ağız: mikrofon → VAD/Smart Turn → Whisper → sesli komut
  kapısı → `TurnSink` (bitmiş sesli tur yönlendiriciye gider); `TTSSpeakFrame` → TTS → hoparlör.
- `src/celal/hermes.py`: `HermesClient`, `/v1/responses` ya da `/v1/chat/completions`, SSE akışı.
- `src/celal/tunnel.py`: `TunnelManager` (ssh -L, yeniden kurma) ve `ReverseTunnel` (ssh -R).
- `src/celal/echo_gate.py`: Celal konuşurken mikrofonu yok sayar; `wake.py`: "Celal dinlemeye başla/durdur".
- `src/celal/ui.py` + `web/index.html`: pywebview penceresi (sekme şeridi); `menubar.py`: rumps simgesi
  (ayrı süreç); `hotkey.py`: Carbon kısayolu.
- Faz 1'den kalanlar: `audio_io.py`, `tts_ema.py`, `tts_say.py`, `stt_filter.py`, `speech_text.py`, `bench.py`.

## Bilinen sınırlar

- Barge-in kapalıyken Celal konuşurken söylediğin duyulmaz; durdurmak için **Durdur**/Esc.
- Whisper parça bazlıdır; konuşman bitince ~1–2 s'de transkript gelir.
- Hermes ilk kelimeyi 5–15 s'de, araç kullanırsa daha geç verir; `celal hermes` ile devredilen iş
  (ör. cron listesi) ~30 s sürebilir.
- Ön eksiz mesaj Hermes sekmesinde de Mac'e gider; Hermes'le sohbete devam etmek için her mesaja
  "sunucuda …" ile başla.
- Mac uykudan dönünce ya da Celal SIGKILL ile ölünce sunucu eski ssh oturumunu ve 8643 portunu tutabilir.
  Celal'in temizlik komutu (`ss -p` ile sshd pid'i bulup öldürme) sunucuda pid'i göremeyebilir (sshd
  süreçleri dump edilemez); o zaman ters tünel "izin vermedi" der, eski oturum TCP keepalive ile (~2 saat)
  ya da elle kapatılınca düzelir.
- 📱 görevlerde `access`/klasör kilidi yok: aynı klasörde iki yazan görev yan yana çalışabilir.

## Lisans

MIT
