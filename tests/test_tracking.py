import unittest

import cv2
import numpy as np

from geometry.homography import estimate_homography, transform_corners
from tracking.tracker import LandingPadTracker
from tracking.recovery import detect_for_recovery, RecoverySchedule
from detection.slow_detector import SlowDetector
from test_detection import render_scene, rotated_quad


class HomographyTrackingTests(unittest.TestCase):
    def test_ransac_rejects_bad_matches(self):
        rng = np.random.default_rng(7)
        before = rng.uniform(40, 500, (70, 2)).astype(np.float32)
        expected = np.float64([[1.01, .02, 7], [-.015, .99, 12], [.00003, -.00002, 1]])
        after = cv2.perspectiveTransform(before[:, None], expected).reshape(-1, 2)
        after += rng.normal(0, .12, after.shape).astype(np.float32)
        after[:15] = rng.uniform(0, 600, (15, 2))
        H, inliers = estimate_homography(before, after)
        self.assertIsNotNone(H)
        self.assertLessEqual(int(inliers[:15].sum()), 1)
        self.assertGreaterEqual(int(inliers[15:].sum()), 53)
        corners = np.float32([[50,50],[450,50],[450,450],[50,450]])
        np.testing.assert_allclose(transform_corners(corners, H),
                                   transform_corners(corners, expected), atol=.5)

    def test_collinear_and_insufficient_matches_fail(self):
        points = np.float32([[i, 2*i] for i in range(12)])
        self.assertIsNone(estimate_homography(points, points+2)[0])
        self.assertIsNone(estimate_homography(points[:3], points[:3])[0])

    def test_transform_preserves_vertex_identity_across_rotation(self):
        corners = np.float32([[10,10],[80,10],[80,60],[10,60]])
        H = np.float64([[0,-1,100],[1,0,0],[0,0,1]])
        np.testing.assert_allclose(transform_corners(corners, H),
                                  [[90,10],[90,80],[40,80],[40,10]])


class RecoveryTests(unittest.TestCase):
    def test_short_disappearance_recovers_before_full_search_deadline(self):
        schedule = RecoverySchedule(30, 5)
        detector = SlowDetector()
        quad = rotated_quad(20)
        visible = render_scene(quad)
        blank = np.full_like(visible, 70)
        first, _, _ = detect_for_recovery(visible, detector)
        self.assertTrue(first.valid)
        schedule.attempted(0, 'full')
        recovered = None
        for index in range(1, 30):
            mode = schedule.due(index, last_valid_frame=0)
            if mode is None:
                continue
            self.assertEqual(mode, 'local')
            result, scope, calls = detect_for_recovery(
                blank if index < 12 else visible, detector, first.corners, allow_global=False)
            self.assertEqual((scope, calls), ('local', 1))
            schedule.attempted(index, mode)
            if index < 12:
                self.assertFalse(result.valid)
                self.assertIsNone(result.corners)
            elif result.valid:
                recovered = index
                break
        self.assertEqual(recovered, 15)

    def test_stale_local_window_waits_for_scheduled_full_fallback(self):
        frame = render_scene(rotated_quad(20))
        stale = np.float32([[20,20],[80,20],[80,100],[20,100]])
        detector = SlowDetector()
        early, scope, calls = detect_for_recovery(frame,detector,stale,allow_global=False)
        self.assertFalse(early.valid)
        self.assertEqual((scope,calls),('local',1))
        scheduled, scope, calls = detect_for_recovery(frame,detector,stale)
        self.assertTrue(scheduled.valid)
        self.assertEqual((scope,calls),('global',2))
        missing, scope, calls = detect_for_recovery(frame,detector,None,allow_global=False)
        self.assertFalse(missing.valid)
        self.assertEqual((scope,calls),('none',0))

    def test_local_retry_burst_expires_and_full_search_cadence_is_preserved(self):
        schedule = RecoverySchedule(30,5)
        attempts = []
        for index in range(61):
            mode = schedule.due(index,last_valid_frame=0)
            if mode is not None:
                attempts.append((index,mode))
                schedule.attempted(index,mode)
        self.assertEqual([i for i,m in attempts if m=='full'],[0,30,60])
        self.assertEqual([i for i,m in attempts if m=='local'],[5,10,15,20,25])


class TrackerTests(unittest.TestCase):
    def test_recovery_restores_coordinates_and_falls_back_from_stale_window(self):
        quad = rotated_quad(20)
        frame = render_scene(quad)
        stale = np.float32([[20,20],[80,20],[80,100],[20,100]])
        for previous, expected_scope, expected_calls in ((quad, 'local', 1),
                                                        (stale, 'global', 2),
                                                        (None, 'global', 1)):
            with self.subTest(scope=expected_scope, calls=expected_calls):
                result, scope, calls = detect_for_recovery(frame, SlowDetector(), previous)
                self.assertTrue(result.valid)
                self.assertEqual((scope, calls), (expected_scope, expected_calls))
                self.assertLess(np.linalg.norm(result.corners-quad, axis=1).max(), 6)

    def test_continuous_motion_and_original_pixel_coordinates(self):
        # Full-resolution input, smaller internal LK image. Ground truth comes
        # from the known warp, independently of the estimated homography.
        quad = rotated_quad(10)*2
        first = render_scene(quad, 'handdrawn_x', canvas=(1920,1440))
        tracker = LandingPadTracker(max_width=960)
        self.assertTrue(tracker.initialize(first, quad))
        for index in range(1, 61):
            angle = .25*index
            matrix = cv2.getRotationMatrix2D((960,720), angle, 1+.001*index)
            matrix[:, 2] += [1.3*index, .8*index]
            H = np.vstack((matrix, [0,0,1]))
            frame = cv2.warpPerspective(first, H, (1920,1440))
            result = tracker.update(frame)
            self.assertTrue(result.valid, (index, result.reason))
            truth = transform_corners(quad, H)
            error = np.linalg.norm(result.corners-truth, axis=1).max()
            self.assertLess(error, 12, (index, error))

    def test_disappearance_clears_state_and_reinitialization_recovers(self):
        quad = rotated_quad(0)
        frame = render_scene(quad, 'handdrawn_x')
        tracker = LandingPadTracker()
        self.assertTrue(tracker.initialize(frame, quad))
        result = tracker.update(np.full_like(frame, 90))
        self.assertFalse(result.valid)
        self.assertIsNone(result.corners)
        self.assertFalse(tracker.initialized)
        self.assertIsNone(tracker.prev_points)
        self.assertTrue(tracker.initialize(frame, quad))
        self.assertTrue(tracker.update(frame).valid)

    def test_perspective_motion_over_stationary_background(self):
        quad = rotated_quad(0)
        tracker = LandingPadTracker()
        self.assertTrue(tracker.initialize(render_scene(quad, 'handdrawn_x'), quad))
        for index in range(1, 16):
            target = quad + np.float32([[.8, .2], [1.1, .6], [.6, .9], [.5, .4]])*index
            result = tracker.update(render_scene(target, 'handdrawn_x'))
            self.assertTrue(result.valid, (index, result.reason))
            self.assertLess(np.linalg.norm(result.corners-target, axis=1).max(), 8)

    def test_blank_pad_cannot_initialize(self):
        tracker = LandingPadTracker()
        self.assertFalse(tracker.initialize(np.full((720,960), 180, np.uint8), rotated_quad(0)))
        self.assertFalse(tracker.update(np.zeros((720,960), np.uint8)).valid)

    def test_covering_a_paper_corner_does_not_keep_a_false_quad(self):
        quad = rotated_quad(0)
        frame = render_scene(quad, 'handdrawn_x')
        tracker = LandingPadTracker()
        self.assertTrue(tracker.initialize(frame, quad))
        covered = frame.copy()
        x, y = np.round(quad[0]).astype(int)
        cv2.rectangle(covered, (x-45, y-45), (x+45, y+45), (80,80,80), -1)
        result = tracker.update(covered)
        self.assertFalse(result.valid)
        self.assertIsNone(result.corners)
        self.assertFalse(tracker.initialized)

    def test_periodic_marker_check_rejects_stable_wrong_target(self):
        # A texture can have perfect LK matches while not being the landing pad.
        # Simulate an incorrect initialization to exercise the drift safeguard.
        quad = rotated_quad(0)
        frame = render_scene(quad, 'checkerboard')
        tracker = LandingPadTracker()
        self.assertTrue(tracker.initialize(frame, quad))
        for _ in range(9):
            self.assertTrue(tracker.update(frame).valid)
        result = tracker.update(frame)
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, 'x_verification_failed')
        self.assertIsNotNone(result.x_check)
        self.assertFalse(result.x_check['valid'])
        self.assertEqual(len(result.x_check['candidate_corners']), 4)
        self.assertFalse(tracker.initialized)
        self.assertIsNone(tracker.update(frame).x_check)

    def test_invalid_input_and_resolution_change_reset(self):
        tracker = LandingPadTracker()
        quad = rotated_quad(0)
        frame = render_scene(quad, 'handdrawn_x')
        for bad in (None, np.zeros((10,10),np.uint8), np.zeros((720,960),np.float32)):
            self.assertFalse(tracker.initialize(bad, quad))
        self.assertTrue(tracker.initialize(frame, quad))
        result = tracker.update(cv2.resize(frame, (480,360)))
        self.assertEqual(result.reason, 'frame_size_changed')
        self.assertFalse(tracker.initialized)


if __name__ == '__main__':
    unittest.main()
