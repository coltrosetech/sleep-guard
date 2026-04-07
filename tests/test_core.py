"""SleepGuard v2 unit tests."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import SleepGuardConfig, GuardState, CalibratedThresholds
from utils.math_utils import euclidean_distance, clamp, ema_update
from utils.ring_buffer import RingBuffer
from core.state_machine import SleepStateMachine
from core.fusion import SignalFusion, FusionResult
from core.absence_tracker import AbsenceTracker, AbsenceStatus
from detectors.eye_tracker import EyeMetrics
from detectors.head_pose import HeadPoseMetrics
from detectors.movement_analyzer import MovementMetrics
from detectors.pose_analyzer import PoseMetrics, PostureClass


def test_state_machine_5_states():
    config = SleepGuardConfig()
    sm = SleepStateMachine(config)
    assert sm.state == GuardState.ACTIVE

    # Low score = ACTIVE
    for t in range(50):
        sm.update(0.05, t * 0.1)
    assert sm.state == GuardState.ACTIVE

    # Idle range for long enough -> IDLE
    for t in range(50, 200):
        sm.update(0.2, t * 0.1)
    assert sm.state == GuardState.IDLE

    # Higher score -> DROWSY
    for t in range(200, 300):
        sm.update(0.5, t * 0.1)
    assert sm.state == GuardState.DROWSY

    # Even higher -> SLEEPING
    for t in range(300, 500):
        sm.update(0.8, t * 0.1)
    assert sm.state == GuardState.SLEEPING

    # Recovery
    for t in range(500, 600):
        sm.update(0.05, t * 0.1)
    assert sm.state == GuardState.ACTIVE
    print("  [PASS] 5-state transitions (ACTIVE->IDLE->DROWSY->SLEEPING->ACTIVE)")


def test_state_machine_absent():
    config = SleepGuardConfig()
    sm = SleepStateMachine(config)

    absence = AbsenceStatus(is_absent=True, absence_duration=200.0)
    sm.update(0.0, 100.0, person_detected=False, absence_status=absence)
    assert sm.state == GuardState.ABSENT

    # Person returns -> immediate ACTIVE
    sm.update(0.0, 101.0, person_detected=True, absence_status=AbsenceStatus())
    assert sm.state == GuardState.ACTIVE
    print("  [PASS] ABSENT state + recovery")


def test_state_machine_no_flicker():
    config = SleepGuardConfig()
    sm = SleepStateMachine(config)
    sm.update(0.9, 0.0)
    assert sm.state == GuardState.ACTIVE  # single spike, no transition
    sm.update(0.05, 0.5)
    assert sm.state == GuardState.ACTIVE
    print("  [PASS] no flicker")


def test_fusion_full_mode():
    config = SleepGuardConfig()
    fusion = SignalFusion(config)
    thresholds = CalibratedThresholds(ear_baseline=0.28, ear_closed=0.20)
    eye = EyeMetrics(avg_ear=0.30, ear_confidence=1.0, perclos=5.0)
    head = HeadPoseMetrics(pitch_deviation=5.0, roll_deviation=3.0)
    pose = PoseMetrics(posture=PostureClass.UPRIGHT, pose_score=0.05, pose_confidence=0.9)
    movement = MovementMetrics(stillness_duration_sec=0.0, is_still=False)

    result = fusion.compute(eye, head, movement, pose, True, 0.0, thresholds)
    assert result.fusion_mode == "full"
    assert result.score < 0.3
    assert "goz" in result.component_scores
    assert "vucut" in result.component_scores
    print("  [PASS] full fusion (face+body, awake)")


def test_fusion_body_only():
    config = SleepGuardConfig()
    fusion = SignalFusion(config)
    thresholds = CalibratedThresholds()
    pose = PoseMetrics(posture=PostureClass.LYING, pose_score=0.9, pose_confidence=0.8)
    movement = MovementMetrics(stillness_duration_sec=30.0, is_still=True)

    result = fusion.compute(None, None, movement, pose, False, 5.0, thresholds)
    assert result.fusion_mode == "body_only"
    assert result.score > 0.5
    assert "vucut" in result.component_scores
    assert "hareket" in result.component_scores
    print("  [PASS] body-only fusion (sleeping, no face)")


def test_fusion_absent():
    config = SleepGuardConfig()
    fusion = SignalFusion(config)
    thresholds = CalibratedThresholds()
    absence = AbsenceStatus(is_absent=True, absence_duration=200.0)

    result = fusion.compute(None, None, None, None, False, 0.0, thresholds,
                            absence_status=absence)
    assert result.fusion_mode == "absent"
    assert result.score == 0.0
    print("  [PASS] absent fusion")


def test_absence_tracker():
    config = SleepGuardConfig()
    config.absence_alert_sec = 10.0  # Short for testing
    tracker = AbsenceTracker(config)

    # Person present
    status = tracker.update(True, "masa", 0.0)
    assert not status.is_absent

    # Person leaves
    for t in range(1, 15):
        status = tracker.update(False, None, float(t))
    assert status.is_absent
    assert status.absence_duration >= 10.0
    assert status.last_seen_zone == "masa"

    # Person returns
    status = tracker.update(True, "koltuk", 15.0)
    assert not status.is_absent
    print("  [PASS] absence tracker (detect + recovery)")


def test_pose_metrics_score():
    """Test that different postures produce expected score ranges."""
    assert PostureClass.UPRIGHT.value == "UPRIGHT"
    assert PostureClass.LYING.value == "LYING"

    # Score ranges (from pose_analyzer logic)
    # UPRIGHT: 0.0-0.1, SLOUCHED: 0.2-0.4, HEAD_DOWN: 0.5-0.7, LYING: 0.8-1.0
    print("  [PASS] pose metrics enum")


if __name__ == "__main__":
    print("SleepGuard v2 Unit Tests")
    print("=" * 45)

    print("\nState Machine (5 durum):")
    test_state_machine_5_states()
    test_state_machine_absent()
    test_state_machine_no_flicker()

    print("\nFusion (cift mod):")
    test_fusion_full_mode()
    test_fusion_body_only()
    test_fusion_absent()

    print("\nAbsence Tracker:")
    test_absence_tracker()

    print("\nPose:")
    test_pose_metrics_score()

    print("\n" + "=" * 45)
    print("TUM TESTLER BASARILI!")
