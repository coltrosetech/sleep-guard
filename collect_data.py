"""45 saniye veri toplama - performans optimize."""
import sys, os, time, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SleepGuardConfig, CalibrationState
from input.video_source import VideoSource
from detectors.person_detector import PersonDetector
from detectors.pose_analyzer import PoseAnalyzer
from detectors.face_detector import FaceDetector
from detectors.eye_tracker import EyeTracker
from detectors.head_pose import HeadPoseEstimator
from detectors.movement_analyzer import MovementAnalyzer
from core.calibrator import AdaptiveCalibrator
from core.fusion import SignalFusion
from core.state_machine import SleepStateMachine
from core.absence_tracker import AbsenceTracker
from core.signal_memory import SignalMemory
import cv2
import numpy as np
from collections import Counter

config = SleepGuardConfig()
config.calibration_duration_sec = 6.0

source = VideoSource(config)
person_det = PersonDetector(config)
pose_ana = PoseAnalyzer(config)
face_det = FaceDetector(config)
eye_track = EyeTracker(config)
head_pose = HeadPoseEstimator(config)
movement = MovementAnalyzer(config)
calibrator = AdaptiveCalibrator(config)
fusion = SignalFusion(config)
sm = SleepStateMachine(config)
absence = AbsenceTracker(config)
sig_mem = SignalMemory(half_life_sec=config.signal_memory_half_life, max_sec=config.signal_memory_max_sec)

out_dir = os.path.join(os.path.dirname(__file__), "data_collection")
os.makedirs(out_dir, exist_ok=True)

data_log = []
start = time.time()
fi = 0
cal_done = False
last_person = None
last_pose = None
POSE_INTERVAL = 3  # pose her 3 frame'de

print("=" * 65)
print("  VERI TOPLAMA - 45 sn | Kamera onunde oturun")
print("=" * 65)

while time.time() - start < 38:
    ret, frame, ts = source.read()
    if not ret:
        break
    fi += 1
    h, w = frame.shape[:2]

    # Isik iyilestirme
    gray_chk = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if gray_chk.mean() < 100:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        frame = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # Pose detection (every N frames - heavy)
    if fi % POSE_INTERVAL == 0:
        person = person_det.process(frame)
        last_person = person
        if person.detected and person.landmarks:
            last_pose = pose_ana.compute(person.landmarks)
    else:
        person = last_person if last_person else type('P', (), {'detected': False, 'landmarks': None, 'estimated_distance': 0, 'bbox_area': 0, 'bbox_center': None})()

    pose_metrics = last_pose
    abs_st = absence.update(person.detected, None, ts)
    body_mov = movement.compute_body(person.landmarks if person.detected else None, frame, ts)

    # Face (every frame when available - lighter than pose)
    eye_m = head_m = None
    face_vis = False
    face_ok = person.detected and person_det.is_face_usable(person.landmarks, person.estimated_distance) if person.detected else False

    if face_ok:
        fr = face_det.process(frame)
        if fr.detected:
            face_vis = True
            th = calibrator.get_thresholds()
            eye_m = eye_track.compute(fr.landmarks, th)
            head_m = head_pose.compute(fr.landmarks, frame.shape)
            head_m = head_pose.compute_deviations(head_m, th)

    # Calibration
    # Arka plan kalibrasyon (engellemez)
    if calibrator.is_refining and person.detected:
        calibrator.feed(eye_m, head_m, body_mov, pose_metrics)
        if not calibrator.is_refining and not cal_done:
            cal_done = True
            t = calibrator.get_thresholds()
            head_pose.set_baseline(t.baseline_pitch, t.baseline_roll)
            print(f"[KAL] EAR={t.ear_baseline:.3f} Govde={t.baseline_torso_angle:.1f} HD={t.baseline_head_drop:.2f}")

    fus = fusion.compute(eye_m, head_m, body_mov, pose_metrics, face_vis,
                         face_det.face_lost_duration, calibrator.get_thresholds(), None, abs_st,
                         signal_memory=sig_mem, timestamp=ts)
    state = sm.update(fus.score, ts, person.detected, abs_st)

    row = {
        "f": fi, "t": round(ts, 2), "person": person.detected,
        "dist": round(person.estimated_distance, 2) if person.detected else None,
        "face_ok": face_ok, "face_vis": face_vis,
        "posture": pose_metrics.posture.value if pose_metrics else None,
        "torso": round(pose_metrics.torso_angle, 1) if pose_metrics else None,
        "hd": round(pose_metrics.head_drop, 3) if pose_metrics else None,
        "p_score": round(pose_metrics.pose_score, 3) if pose_metrics else None,
        "p_conf": round(pose_metrics.pose_confidence, 2) if pose_metrics else None,
        "ear": round(eye_m.avg_ear, 4) if eye_m else None,
        "ear_c": round(eye_m.ear_confidence, 2) if eye_m else None,
        "perc": round(eye_m.perclos, 1) if eye_m else None,
        "p_dev": round(head_m.pitch_deviation, 1) if head_m else None,
        "r_dev": round(head_m.roll_deviation, 1) if head_m else None,
        "disp": round(body_mov.displacement, 5),
        "still": round(body_mov.stillness_duration_sec, 1),
        "score": round(fus.score, 3), "mode": fus.fusion_mode, "state": state.value,
    }
    data_log.append(row)

    # Snapshot
    if fi % 30 == 0 and person.detected and person.landmarks:
        s = frame.copy()
        for lm in person.landmarks:
            if lm.visibility > 0.3:
                cv2.circle(s, (int(lm.x * w), int(lm.y * h)), 3, (0, 255, 0), -1)
        cv2.imwrite(os.path.join(out_dir, f"s_{fi:04d}.jpg"), s)

    if fi % 8 == 0:
        p = f"G={pose_metrics.torso_angle:.0f} {pose_metrics.posture.value[:4]} HD={pose_metrics.head_drop:.2f}" if pose_metrics else "poz-yok"
        e = f"EAR={eye_m.avg_ear:.3f}" if eye_m else "goz-yok"
        d = f"{person.estimated_distance:.1f}m" if person.detected else "---"
        print(f"[{ts:5.1f}s] {state.value:10s} {fus.score:.2f} {fus.fusion_mode:9s} d={d} | {p} | {e} | disp={body_mov.displacement:.4f}")

elapsed = time.time() - start
source.release()
person_det.release()
face_det.release()

csv_path = os.path.join(out_dir, "log.csv")
if data_log:
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=data_log[0].keys())
        w.writeheader()
        w.writerows(data_log)

print("\n" + "=" * 65)
print("  OZET")
print("=" * 65)
print(f"  {fi} frame | {elapsed:.1f}s | {fi/elapsed:.1f} FPS | {len(data_log)} veri")
if data_log:
    det = sum(1 for r in data_log if r["person"])
    fv = sum(1 for r in data_log if r["face_vis"])
    print(f"  Kisi: {det}/{len(data_log)} ({det/len(data_log)*100:.0f}%) | Yuz: {fv}/{len(data_log)} ({fv/len(data_log)*100:.0f}%)")
    poses = Counter([r["posture"] for r in data_log if r["posture"]])
    print(f"  Poz: {dict(poses)}")
    states = Counter([r["state"] for r in data_log])
    print(f"  Durum: {dict(states)}")
    ears = [r["ear"] for r in data_log if r["ear"]]
    if ears: print(f"  EAR: {min(ears):.3f}-{max(ears):.3f} ort={np.mean(ears):.3f}")
    angs = [r["torso"] for r in data_log if r["torso"] is not None]
    if angs: print(f"  Govde: {min(angs):.0f}-{max(angs):.0f} ort={np.mean(angs):.0f}")
    hds = [r["hd"] for r in data_log if r["hd"] is not None]
    if hds: print(f"  BasDusme: {min(hds):.2f}-{max(hds):.2f} ort={np.mean(hds):.2f}")
    fs = [r["score"] for r in data_log]
    print(f"  Skor: {min(fs):.3f}-{max(fs):.3f} ort={np.mean(fs):.3f}")
    modes = Counter([r["mode"] for r in data_log])
    print(f"  Mod: {dict(modes)}")
    dists = [r["dist"] for r in data_log if r["dist"]]
    if dists: print(f"  Mesafe: {min(dists):.1f}m-{max(dists):.1f}m ort={np.mean(dists):.1f}m")
print(f"  CSV: {csv_path}")
print("=" * 65)
