"""
SleepGuard v2 - Guvenlik Kabini Uyku Tespit Sistemi
====================================================

Ozellikler:
  - Tam vucut tespiti (MediaPipe Pose - 33 nokta)
  - Yuz analizi (yakin mesafede otomatik)
  - Bolge yonetimi (masa/koltuk/kapi)
  - Yokluk tespiti (3dk alan terk)
  - Hareketsizlik tespiti (5dk IDLE)
  - 5 durum: AKTIF / HAREKETSIZ / UYUKLAMA / UYUYOR / ALAN TERK

Kullanim:
  python main.py                          # Canli kamera
  python main.py --video kayit.mp4        # Video dosyasindan
  python main.py --setup-zones            # Interaktif bolge tanimlama
  python main.py --calibration-time 10    # Kisa kalibrasyon
  python main.py --no-sound               # Sessiz mod
  python main.py --no-face                # Sadece vucut (yuz analizi yok)

Klavye:
  Q/ESC : Cikis
  R     : Yeniden kalibrasyon
  S     : Iskelet goster/gizle
  D     : Debug panel goster/gizle
  M     : Yuz mesh goster/gizle
  Z     : Bolge goster/gizle
  P     : Duraklat
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SleepGuardConfig
from core.pipeline import DetectionPipeline
from utils.logger import setup_logger

logger = setup_logger()


def main():
    parser = argparse.ArgumentParser(
        description="SleepGuard v2 - Guvenlik Kabini Uyku Tespit Sistemi",
    )
    parser.add_argument("--video", type=str, default=None,
                        help="Video dosyasi yolu")
    parser.add_argument("--camera", type=int, default=0,
                        help="Kamera indeksi (varsayilan: 0)")
    parser.add_argument("--calibration-time", type=float, default=30.0,
                        help="Kalibrasyon suresi (sn)")
    parser.add_argument("--no-sound", action="store_true",
                        help="Sesi kapat")
    parser.add_argument("--no-face", action="store_true",
                        help="Yuz analizi devre disi (sadece vucut)")
    parser.add_argument("--no-debug", action="store_true",
                        help="Debug panelini gizle")
    parser.add_argument("--setup-zones", action="store_true",
                        help="Interaktif bolge tanimlama")
    parser.add_argument("--zones-file", type=str, default="zones.json",
                        help="Bolge tanimlari dosyasi")
    parser.add_argument("--absence-time", type=float, default=180.0,
                        help="Yokluk alarm suresi (sn, varsayilan: 180)")
    parser.add_argument("--idle-time", type=float, default=300.0,
                        help="Hareketsizlik alarm suresi (sn, varsayilan: 300)")

    args = parser.parse_args()

    config = SleepGuardConfig()
    config.video_path = args.video
    config.camera_index = args.camera
    config.calibration_duration_sec = args.calibration_time
    config.sound_enabled = not args.no_sound
    config.show_debug = not args.no_debug
    config.zones_file = args.zones_file
    config.absence_alert_sec = args.absence_time
    config.idle_threshold_sec = args.idle_time

    if args.no_face:
        config.face_distance_max = 0.0  # Never use face

    logger.info("=" * 55)
    logger.info("  SleepGuard v2 - Guvenlik Kabini Uyku Tespit")
    logger.info("=" * 55)
    src = f"Video: {config.video_path}" if config.video_path else f"Kamera: {config.camera_index}"
    logger.info(f"  Kaynak: {src}")
    logger.info(f"  Kalibrasyon: {config.calibration_duration_sec}s")
    logger.info(f"  Yokluk alarmi: {config.absence_alert_sec}s")
    logger.info(f"  Hareketsizlik: {config.idle_threshold_sec}s")
    logger.info(f"  Yuz analizi: {'Kapali' if args.no_face else 'Otomatik (yakin mesafe)'}")
    logger.info(f"  Ses: {'Acik' if config.sound_enabled else 'Kapali'}")
    logger.info("=" * 55)

    pipeline = DetectionPipeline(config)

    # Interactive zone setup
    if args.setup_zones:
        from input.video_source import VideoSource
        logger.info("Bolge tanimlama modu...")
        ret, frame, _ = pipeline.source.read()
        if ret:
            pipeline.zone_manager.setup_interactive(frame)

    pipeline.run()


if __name__ == "__main__":
    main()
