"""Visible drift must remain evidence; missing endpoints cannot yield a negative."""
import unittest
from app.befast.arms import ArmDriftScreen
from app.befast.config import BefastConfig
from test_befast import standing_points, make_keypoints


def hold_trial(mode):
    s = ArmDriftScreen(BefastConfig())
    # Isolate the hold/return measurement contract after successful raised entry.
    s.start(0); s.phase_state = 'recording'; s.phase_index = 1
    s.phase_state_started_at = 0
    for i in range(51):
        t = i / 10
        y = .30
        if mode == 'sudden' and t > 3: y = .45
        if mode == 'gradual': y += .15 * t / 5
        points = standing_points(left_wrist_y=y)
        if (mode == 'late_occlusion' and t > 3) or (mode == 'brief_occlusion' and 2 < t < 2.5):
            points = make_keypoints()
        s.update(t, points)
    for t in (5.1, 5.6):
        s.update(t, standing_points(left_wrist_y=.55, right_wrist_y=.55))
    return s.finish()


class ArmMeasurementRetentionTest(unittest.TestCase):
    def test_visible_sudden_lowering_is_retained(self):
        r = hold_trial('sudden')
        self.assertEqual(r.status, 'positive')
        self.assertGreater(r.metrics['left_drop'], .7)

    def test_visible_gradual_lowering_is_retained(self):
        self.assertEqual(hold_trial('gradual').status, 'positive')

    def test_missing_late_endpoint_is_incomplete(self):
        r = hold_trial('late_occlusion')
        self.assertEqual(r.status, 'insufficient')
        self.assertEqual(r.reason, 'arm_hold_endpoint_not_visible')

    def test_brief_middle_occlusion_preserves_observed_negative(self):
        self.assertEqual(hold_trial('brief_occlusion').status, 'negative')

    def test_stable_visible_hold_is_negative(self):
        self.assertEqual(hold_trial('stable').status, 'negative')
