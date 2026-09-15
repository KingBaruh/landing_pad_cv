import json
import unittest

import cv2
import numpy as np

from geometry.homography import order_corners
from geometry.pose import A4_OBJECT_POINTS, estimate_pose, rotation_to_rpy
from tracking.demo import pose_record
from detection.slow_detector import SlowDetector
from tracking.recovery import detect_for_recovery
from test_detection import render_scene, corner_error


def rotation(roll, pitch, yaw):
    cx, cy, cz = np.cos([roll, pitch, yaw])
    sx, sy, sz = np.sin([roll, pitch, yaw])
    return (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @
            np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]) @
            np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]))


class PoseTests(unittest.TestCase):
    K = np.float64([[1350, 0, 960], [0, 1355, 540], [0, 0, 1]])

    def project(self, R, xyz, dist=None):
        return cv2.projectPoints(A4_OBJECT_POINTS, cv2.Rodrigues(R)[0],
                                 np.float64(xyz), self.K, dist)[0].reshape(4, 2)

    def assert_rotation_mod_half_turn(self, pose, expected, tolerance=1e-5):
        actual = cv2.Rodrigues(pose.rvec)[0]
        self.assertLess(min(np.linalg.norm(actual-expected),
                            np.linalg.norm(actual-expected @ np.diag([-1, -1, 1]))), tolerance)

    def test_metric_pose_with_rotations_and_all_cyclic_corner_orders(self):
        xyz = np.array([.08, -.04, .85])
        for yaw in (0, 60, 95, 170, 250):
            R = rotation(.3, -.4, np.deg2rad(yaw))
            corners = self.project(R, xyz)
            for reverse in (False, True):
                for shift in range(4):
                    with self.subTest(yaw=yaw, reverse=reverse, shift=shift):
                        inputs = np.roll(corners[::-1] if reverse else corners, shift, axis=0)
                        pose = estimate_pose(inputs, self.K, None)
                        self.assertTrue(pose.valid, pose.reason)
                        np.testing.assert_allclose(pose.position_xyz, xyz, atol=1e-6)
                        self.assertAlmostEqual(pose.distance_m, np.linalg.norm(xyz), places=6)
                        self.assert_rotation_mod_half_turn(pose, R)
                        projected = self.project(cv2.Rodrigues(pose.rvec)[0], pose.tvec)
                        np.testing.assert_allclose(projected, inputs[pose.corner_indices], atol=1e-5)
                        self.assertTrue(pose.yaw_ambiguous_180)

    def test_fronto_parallel_including_optical_axis(self):
        for xyz in ([0, 0, .4], [0, 0, 1.], [.15, -.2, 3.]):
            pose = estimate_pose(self.project(np.eye(3), xyz), self.K, np.zeros(5))
            self.assertTrue(pose.valid)
            np.testing.assert_allclose(pose.position_xyz, xyz, atol=1e-6)
            self.assertGreaterEqual(pose.distance_m, pose.position_xyz[2]-1e-12)

    def test_distorted_and_undistorted_coordinates_give_same_metric_pose(self):
        dist = np.float64([.12, -.04, .001, -.002, .01])
        R, xyz = rotation(.2, -.3, .4), [.2, .1, .9]
        raw = self.project(R, xyz, dist)
        corrected = cv2.undistortPoints(raw[:, None], self.K, dist, P=self.K).reshape(4, 2)
        for points, coefficients in ((raw, dist), (corrected, None)):
            pose = estimate_pose(points, self.K, coefficients)
            self.assertTrue(pose.valid)
            np.testing.assert_allclose(pose.position_xyz, xyz, atol=1e-5)
            self.assert_rotation_mod_half_turn(pose, R, tolerance=1e-4)

    def test_subpixel_corner_noise(self):
        rng = np.random.default_rng(41)
        R, xyz = rotation(.35, -.3, .1), np.array([.06, -.03, 1.2])
        points = self.project(R, xyz)
        for _ in range(20):
            pose = estimate_pose(points+rng.normal(0, .2, (4, 2)), self.K, None)
            self.assertTrue(pose.valid)
            self.assertLess(np.linalg.norm(pose.position_xyz-xyz), .015)
            self.assert_rotation_mod_half_turn(pose, R, tolerance=.06)
            self.assertLess(pose.reprojection_error_px, 1.)

    def test_continuity_across_image_corner_reordering_and_yaw_90(self):
        previous = None
        for yaw in range(80, 111):
            R = rotation(.2, -.3, np.deg2rad(yaw))
            # The detector chooses the image's top-left; its index changes as
            # the pad turns. This is not a physical corner identity.
            corners = order_corners(self.project(R, [.05, .02, 1.]))
            pose = estimate_pose(corners, self.K, None, previous=previous)
            self.assertTrue(pose.valid)
            if previous is not None:
                old_R = cv2.Rodrigues(previous.rvec)[0]
                new_R = cv2.Rodrigues(pose.rvec)[0]
                self.assertLess(np.linalg.norm(new_R-old_R), .03)
            previous = pose

    def test_weak_perspective_reports_alternate_orientation(self):
        corners = self.project(rotation(.15, .2, .1), [0, 0, 8.])
        pose = estimate_pose(corners, self.K, None)
        self.assertTrue(pose.valid)
        self.assertTrue(pose.orientation_ambiguous)

    def test_bad_geometry_and_loss_do_not_return_stale_pose(self):
        corners = self.project(rotation(.3, -.2, .1), [0, 0, 1.])
        previous = estimate_pose(corners, self.K, None)
        invalid_inputs = [None, np.zeros((4, 2)), corners[[0, 2, 1, 3]],
                          [[0, 0], [1, 1], [2, 2], [3, 3]],
                          np.full((4, 2), np.nan), corners[:3]]
        for points in invalid_inputs:
            pose = estimate_pose(points, self.K, None, previous=previous)
            self.assertFalse(pose.valid)
            self.assertIsNone(pose.position_xyz)
            self.assertIsNone(pose.distance_m)
            self.assertIsNone(pose.rvec)
            json.dumps(pose_record(pose), allow_nan=False)
        corners[2] += [80, -40]
        pose = estimate_pose(corners, self.K, None)
        self.assertFalse(pose.valid)
        self.assertEqual(pose.reason, 'high_reprojection_error')
        self.assertGreater(pose.reprojection_error_px, 5.)

    def test_bad_calibration_and_threshold_fail_explicitly(self):
        points = self.project(np.eye(3), [0, 0, 1.])
        for K in (np.zeros((3, 3)), np.eye(2), np.full((3, 3), np.nan)):
            with self.assertRaises(ValueError):
                estimate_pose(points, K, None)
        for dist in ([1, 2, 3], [0, 0, 0, np.nan]):
            with self.assertRaises(ValueError):
                estimate_pose(points, self.K, dist)
        for threshold in (0, -1, np.nan):
            with self.assertRaises(ValueError):
                estimate_pose(points, self.K, None, max_reprojection_error_px=threshold)

    def test_euler_convention_and_json_units(self):
        for pitch in (.2, np.pi/2, -np.pi/2):
            R = rotation(.3, pitch, -.4)
            np.testing.assert_allclose(rotation(*rotation_to_rpy(R)), R, atol=1e-8)
        R = rotation(.3, .2, -.4)
        pose = estimate_pose(self.project(R, [.1, .2, 1.]), self.K, None)
        record = json.loads(json.dumps(pose_record(pose), allow_nan=False))
        np.testing.assert_allclose(record['orientation_rpy_deg'], np.rad2deg([.3, .2, -.4]), atol=1e-6)
        np.testing.assert_allclose(record['position_xyz_m'], [.1, .2, 1.], atol=1e-6)

    def test_calibrated_detection_rejects_large_non_a4_x_decoy(self):
        K = np.float64([[900,0,640],[0,900,360],[0,0,1]])
        R = rotation(.25, -.3, .15)
        quad = cv2.projectPoints(A4_OBJECT_POINTS, cv2.Rodrigues(R)[0],
                                 np.float64([.25,0,1.]), K, None)[0].reshape(4,2)
        decoy = np.float32([[70,80],[480,80],[480,510],[70,510]])
        frame = render_scene(decoy, canvas=(1280,720))
        actual = render_scene(quad, canvas=(1280,720))
        mask = np.zeros(frame.shape[:2], np.uint8)
        cv2.fillConvexPoly(mask, np.round(quad).astype(np.int32), 255)
        frame[mask > 0] = actual[mask > 0]
        detector = SlowDetector(camera_matrix=K)
        detection = detector.detect(frame)
        self.assertTrue(detection.valid)
        self.assertLess(corner_error(detection.corners, quad), 4.)
        self.assertGreater(detector.last_pose_rejections, 0)

    def test_calibrated_local_recovery_uses_full_frame_pixel_coordinates(self):
        K = np.float64([[900,0,640],[0,900,360],[0,0,1]])
        R = rotation(.3,-.4,.2)
        xyz = np.float64([.28,-.12,1.05])
        quad = cv2.projectPoints(A4_OBJECT_POINTS, cv2.Rodrigues(R)[0], xyz,K,None)[0].reshape(4,2)
        frame = render_scene(quad, canvas=(1280,720))
        result, scope, calls = detect_for_recovery(frame, SlowDetector(camera_matrix=K), quad)
        self.assertTrue(result.valid)
        self.assertEqual((scope,calls), ('local',1))
        pose = estimate_pose(result.corners,K,None)
        self.assertTrue(pose.valid)
        np.testing.assert_allclose(pose.position_xyz,xyz,atol=.015)

    def test_thin_bright_background_bridges_do_not_expand_paper_corners(self):
        K = np.float64([[900,0,640],[0,900,360],[0,0,1]])
        quad = cv2.projectPoints(A4_OBJECT_POINTS,np.zeros(3),np.float64([0,0,.8]),K,None)[0].reshape(4,2)
        frame = render_scene(quad, canvas=(1280,720))
        # Bright tabletop veins touch the paper along its left boundary.
        left, top = np.round(quad[0]).astype(int)
        for y in range(top+10, int(quad[3,1])-10, 19):
            cv2.line(frame, (left-30,y-6), (left,y), (245,245,245), 3)
        result = SlowDetector(camera_matrix=K).detect(frame)
        self.assertTrue(result.valid)
        self.assertLess(corner_error(result.corners,quad),4.)


if __name__ == '__main__':
    unittest.main()
