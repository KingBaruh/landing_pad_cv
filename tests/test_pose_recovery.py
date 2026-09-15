"""Integration regression: plausible tracking must not indefinitely hide bad pose."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from geometry.pose import A4_OBJECT_POINTS
from tracking.demo import main
from tracking.tracker import LandingPadTracker
from test_detection import render_scene
from test_pose import rotation


class PoseRecoveryTests(unittest.TestCase):
    def test_transient_rejection_preserves_track_sustained_drift_reacquires(self):
        root = Path('outputs/test_tmp')
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as temporary:
            output = Path(temporary)
            video = output / 'drift.avi'
            calibration = output / 'camera.npz'
            K = np.float64([[900,0,640],[0,900,360],[0,0,1]])
            np.savez(calibration, camera_matrix=K, dist_coeffs=np.zeros(5), image_size=[1280,720])
            corners = cv2.projectPoints(A4_OBJECT_POINTS, cv2.Rodrigues(rotation(.3,-.35,.2))[0],
                                       np.float64([.01,-.02,.85]), K, None)[0].reshape(4,2)
            frame = render_scene(corners, canvas=(1280,720))
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 30, (1280,720))
            self.assertTrue(writer.isOpened())
            try:
                for _ in range(10): writer.write(frame)
            finally:
                writer.release()
            original_update = LandingPadTracker.update
            updates = 0
            def inject_drift(tracker, image):
                nonlocal updates
                result = original_update(tracker, image)
                updates += 1
                if result.valid and updates in (1,2,4,5,6):
                    # Keep the image/solver real; inject a wrong corner into an
                    # otherwise successful tracker, as in the observed failure.
                    tracker.corners[0,0] += 60
                    result.corners = tracker.corners.copy()
                return result
            with patch.object(LandingPadTracker, 'update', inject_drift), \
                 patch.object(sys, 'argv', ['tracking.demo','--video',str(video),
                              '--camera-params',str(calibration),'--pose','--output',str(output)]), \
                 contextlib.redirect_stdout(io.StringIO()):
                main()
            report = json.loads((output/'drift/results.json').read_text())
            rows = report['frames']
            for i in (1,2,4,5):
                self.assertTrue(rows[i]['valid'])
                self.assertFalse(rows[i]['pose']['valid'])
                self.assertIsNone(rows[i]['pose']['position_xyz_m'])
            self.assertTrue(rows[3]['pose']['valid'])
            self.assertEqual(rows[3]['pose_rejection_streak'], 0)
            self.assertEqual(rows[6]['tracking_failure'], 'pose_inconsistent')
            self.assertFalse(rows[6]['valid'])
            self.assertIsNone(rows[6]['corners'])
            self.assertIsNotNone(rows[6]['rejected_tracking_corners'])
            self.assertFalse(rows[6]['pose']['valid'])
            self.assertEqual(rows[7]['source'], 'detector')
            self.assertTrue(rows[7]['pose']['valid'])
            self.assertEqual(report['pose_triggered_resets'], 1)


if __name__ == '__main__':
    unittest.main()
