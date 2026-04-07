# SleepGuard v2

Guvenlik kabinleri icin gercek zamanli uyku ve dikkat tespit sistemi. MediaPipe tabanli vucut ve yuz analizi ile 5 farkli durum tespiti yapar.

## Ozellikler

- **Tam vucut tespiti** - MediaPipe Pose (33 iskelet noktasi), 5 kisiye kadar coklu tespit
- **Yuz analizi** - Yakin mesafede otomatik aktif, 478 landmark ile goz/bas takibi
- **EAR & PERCLOS** - Goz kapanma orani ve 60 saniyelik kapali goz yuzdesi
- **Bas pozisyonu** - solvePnP ile 3D pitch/roll/yaw acilari
- **Vucut pozturu** - Dik, cokmis, bas dusuk, yatma siniflandirmasi
- **Hareket analizi** - Iskelet tabanli hareket + frame-diff fallback
- **Bolge yonetimi** - Masa, koltuk, kapi icin poligon tabanli alan tanimlama
- **Yokluk tespiti** - 3 dakika kisi gorulmezse ALAN TERK alarmi
- **Hareketsizlik tespiti** - 5 dakika hareketsizlikte IDLE alarmi
- **Sinyal hafizasi** - Yuz kayboldiginda son bilinen degerleri 15 saniyelik azalan hafizayla korur
- **Uyarlanabilir kalibrasyon** - Sistem aninda calisir, 30 saniyede arka planda esikleri ayarlar
- **Sesli uyari** - Durum bazli farklilasan bip desenleri

## Durumlar

| Durum | Skor Esigi | Aciklama |
|---|---|---|
| AKTIF | < 0.15 | Normal, uyanik |
| HAREKETSIZ | 0.15 - 0.40 | Uzun suredir hareket yok |
| UYUKLAMA | 0.40 - 0.60 | Gozler kapaniyor, bas dusuyor |
| UYUYOR | > 0.60 | Uyku tespiti, alarm |
| ALAN TERK | - | 3 dakika kisi yok |

## Uyari Seviyeleri

```
NONE → INFO → WARNING → ALARM → CRITICAL → ABSENCE
```

- **WARNING**: Uyuklama basladiktan sonra
- **ALARM**: Uyku tespitinde
- **CRITICAL**: Uyku 15 saniyeyi asarsa yukseltme
- **ABSENCE**: Alan terk durumunda

## Mimari

```
Kamera/Video
    │
    ▼
PersonDetector (MediaPipe Pose - 33 nokta)
    │
    ├── PoseAnalyzer ──────────── vucut posturu skoru
    ├── MovementAnalyzer ──────── hareket/hareketsizlik
    │
    ├── FaceDetector (yakin mesafede otomatik)
    │     ├── EyeTracker ──────── EAR + PERCLOS
    │     └── HeadPoseEstimator ─ pitch/roll/yaw
    │
    ├── AdaptiveCalibrator ─── arka plan esik ayarlama
    └── ZoneManager ───────── bolge konteksti
            │
            ▼
      SignalFusion (3 mod: full / memory / body_only)
            │
            ▼
      SleepStateMachine (5 durum, histerezis korumali)
            │
            ▼
      AlertManager → Sesli uyari
            │
            ▼
      OverlayRenderer → Ekran goruntusu (HUD)
```

### Fuzyon Modlari

| Mod | Kosul | Agirliklar |
|---|---|---|
| **full** | Yuz gorunuyor | Vucut %25, EAR %25, PERCLOS %15, Bas %15, Hareket %15 |
| **memory** | Yuz yeni kayboldu | Onbellekteki goz sinyalleri (azalan) + canli vucut |
| **body_only** | Yuz yok / uzak | Vucut %40, Hareket %35, Bolge %15 |

## Kurulum

### Gereksinimler

- Python 3.10+
- Web kamera veya video dosyasi

### Bagimliliklari Yukle

```bash
pip install -r requirements.txt
```

Bagimlilklar:
- `opencv-python` >= 4.8.0
- `mediapipe` >= 0.10.9
- `numpy` >= 1.24.0
- `scipy` >= 1.10.0

## Kullanim

### Temel Kullanim

```bash
# Canli kamera ile
python main.py

# Video dosyasindan
python main.py --video kayit.mp4

# Belirli kamera indeksi
python main.py --camera 1
```

### Bolge Tanimlama

```bash
# Interaktif bolge tanimlama (masa, koltuk, kapi)
python main.py --setup-zones

# Ozel bolge dosyasi
python main.py --zones-file ozel_bolgeler.json
```

### Ayar Secenekleri

```bash
# Kisa kalibrasyon (test icin)
python main.py --calibration-time 10

# Sessiz mod
python main.py --no-sound

# Sadece vucut analizi (yuz devre disi)
python main.py --no-face

# Debug panelini gizle
python main.py --no-debug

# Yokluk alarm suresi (saniye)
python main.py --absence-time 120

# Hareketsizlik alarm suresi (saniye)
python main.py --idle-time 600
```

### Ornek Kombinasyonlar

```bash
# Test: kisa kalibrasyon, sessiz, video
python main.py --video test.mp4 --calibration-time 5 --no-sound

# Genis kabin: uzun yokluk suresi, sadece vucut
python main.py --no-face --absence-time 300

# Uretim: tam ozellik
python main.py --absence-time 180 --idle-time 300
```

## Klavye Kontrolleri

| Tus | Islem |
|---|---|
| `Q` / `ESC` | Cikis |
| `R` | Yeniden kalibrasyon |
| `S` | Iskelet goster/gizle |
| `D` | Debug panel goster/gizle |
| `M` | Yuz mesh goster/gizle |
| `Z` | Bolge goster/gizle |
| `P` | Duraklat |

## Proje Yapisi

```
SleepGuard/
├── main.py                  # Ana giris noktasi ve CLI
├── config.py                # Konfigrasyon ve esik degerleri
├── collect_data.py          # Benchmark veri toplama araci
├── requirements.txt         # Python bagimliliklari
│
├── core/                    # Cekirdek karar mekanizmasi
│   ├── pipeline.py          # Ana tespit dongusu
│   ├── calibrator.py        # Uyarlanabilir kalibrasyon
│   ├── fusion.py            # 3 modlu sinyal fuzyonu
│   ├── state_machine.py     # 5 durumlu sonlu durum makinesi
│   ├── zone_manager.py      # Bolge yonetimi
│   ├── absence_tracker.py   # Yokluk takibi
│   └── signal_memory.py     # Ustel azalan sinyal hafizasi
│
├── detectors/               # Bilgisayarli goru modulleri
│   ├── person_detector.py   # MediaPipe Pose (33 nokta)
│   ├── pose_analyzer.py     # Vucut posturu siniflandirma
│   ├── face_detector.py     # MediaPipe Face (478 landmark)
│   ├── eye_tracker.py       # EAR ve PERCLOS hesaplama
│   ├── head_pose.py         # 3D bas poz tahmini
│   └── movement_analyzer.py # Hareket ve hareketsizlik analizi
│
├── alert/                   # Uyari sistemi
│   ├── alert_manager.py     # Durum-uyari eslemesi
│   └── sound.py             # Sesli uyari (winsound)
│
├── ui/                      # Kullanici arayuzu
│   ├── overlay.py           # HUD ve gorsel katmanlar
│   └── display.py           # Pencere yonetimi ve FPS
│
├── input/                   # Video girisi
│   └── video_source.py      # Kamera/dosya birlestirici
│
├── utils/                   # Yardimci araclar
│   ├── logger.py            # Yapilandirilmis loglama
│   ├── math_utils.py        # Mesafe, normalizasyon, EMA
│   └── ring_buffer.py       # Dairesel tampon (istatistik)
│
└── data_collection/         # Veri toplama
    └── collector.py         # CSV ciktili metrik toplama
```

## Veri Toplama

Benchmark ve test icin veri toplama araci:

```bash
python collect_data.py
```

38 saniyelik bir oturum boyunca tum metrikleri CSV olarak kaydeder ve ozet istatistikler uretir.

## Konfigrasyon Detaylari

Tum esikler `config.py` icindeki `SleepGuardConfig` sinifinda tanimlidir:

| Parametre | Varsayilan | Aciklama |
|---|---|---|
| `calibration_duration_sec` | 30.0 | Kalibrasyon suresi |
| `ear_closed_ratio` | 0.72 | Goz kapali EAR orani |
| `ear_drowsy_ratio` | 0.82 | Uyuklama EAR orani |
| `head_pitch_drowsy` | 20.0° | Bas egimi uyuklama esigi |
| `head_pitch_sleeping` | 35.0° | Bas egimi uyku esigi |
| `torso_upright_max` | 15.0° | Dik oturma limiti |
| `torso_slouch_max` | 40.0° | Cokmis oturma limiti |
| `torso_lying_min` | 60.0° | Yatma baslangic acisi |
| `face_distance_max` | 1.5m | Yuz analizi menzili |
| `absence_alert_sec` | 180s | Yokluk alarm suresi |
| `idle_threshold_sec` | 300s | Hareketsizlik alarm suresi |
| `signal_memory_half_life` | 15.0s | Sinyal hafiza yari omru |

## Hedef Donanim

| Ortam | Donanim | Notlar |
|---|---|---|
| Gelistirme | PC + USB kamera | Tam hiz, debug modu |
| Uretim | Jetson Nano + USB kamera | Edge inference, dusuik guc tuketimi |

Jetson Nano uzerinde calisirken internet baglantisi sadece alarm bildirimi icin gereklidir. Tum tespit islemleri yerel olarak gerceklesir.

## Lisans

Bu proje ozel kullanim icin gelistirilmistir.
