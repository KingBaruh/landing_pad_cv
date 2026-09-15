"""Run with: python -m unittest discover -s tests -v"""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from calibration.calibrate import calibrate_camera, load_camera_params, save_camera_params


class CalibrationTests(unittest.TestCase):
    def test_recovers_known_camera_from_rendered_checkerboards(self):
        # Independently render 12 board poses through a known pinhole camera.
        # This exercises real corner detection, refinement and calibration.
        camera = np.array([[820., 0., 480.], [0., 810., 360.], [0., 0., 1.]])
        rng = np.random.default_rng(17)
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "images"
            folder.mkdir()
            for view in range(12):
                frame = np.full((720, 960), 180, np.uint8)
                rotation = rng.uniform(-0.4, 0.4, 3)
                translation = np.array([rng.uniform(-0.14, -0.06),
                                        rng.uniform(-0.10, -0.03),
                                        rng.uniform(0.48, 0.68)])
                for row in range(7):
                    for col in range(10):
                        square = np.array([[col-1, row-1, 0], [col, row-1, 0],
                                           [col, row, 0], [col-1, row, 0]], np.float32) * 0.025
                        pixels, _ = cv2.projectPoints(square, rotation, translation,
                                                     camera, np.zeros(5))
                        cv2.fillConvexPoly(frame, np.round(pixels[:, 0] * 16).astype(np.int32),
                                           255 if (row + col) % 2 else 0,
                                           lineType=cv2.LINE_AA, shift=4)
                cv2.imwrite(str(folder / f"{view:02}.png"), frame)
            (folder / "empty.jpg").touch()
            debug = Path(temp) / "debug"
            with contextlib.redirect_stdout(io.StringIO()):
                result = calibrate_camera(str(folder), square_size=0.025, debug_dir=debug,
                                          max_detection_width=480, radial_order=1)
            self.assertEqual(len(result["used_images"]), 12)
            self.assertEqual(len(result["skipped_images"]), 1)
            self.assertLess(result["rms_px"], 0.6)
            self.assertEqual(result["radial_order"], 1)
            np.testing.assert_array_equal(result["dist_coeffs"].ravel()[[1, 4]], [0., 0.])
            np.testing.assert_allclose(result["camera_matrix"][[0, 1], [0, 1]],
                                       camera[[0, 1], [0, 1]], rtol=0.04)
            np.testing.assert_allclose(result["camera_matrix"][:2, 2], camera[:2, 2], atol=12)
            self.assertEqual(len(list(debug.glob("*_corners.png"))), 12)
            self.assertTrue((debug / "undistortion_before_after.png").is_file())
            output = Path(temp) / "saved" / "camera.npz"
            save_camera_params(str(output), **result)
            loaded = load_camera_params(str(output))
            for key in result:
                np.testing.assert_array_equal(loaded[key], result[key])

    def test_rejects_missing_empty_and_mixed_resolution_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            with self.assertRaises(FileNotFoundError):
                calibrate_camera(str(folder / "missing"))
            with self.assertRaisesRegex(ValueError, "No supported"):
                calibrate_camera(temp)
            cv2.imwrite(str(folder / "a.png"), np.full((200, 300), 255, np.uint8))
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "Only 0 usable"):
                    calibrate_camera(temp)
                cv2.imwrite(str(folder / "b.png"), np.full((300, 400), 255, np.uint8))
                with self.assertRaisesRegex(ValueError, "Mixed image resolutions"):
                    calibrate_camera(temp)


if __name__ == "__main__":
    unittest.main()
