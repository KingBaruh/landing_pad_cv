"""Preparation must preserve resized pixels, timestamps and calibration geometry."""
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from common.prepared_video import prepare_video


class PreparedVideoTests(unittest.TestCase):
    def test_lossless_resize_all_frames_fps_and_cache_reuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/'source.avi'
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'FFV1'), 30., (160, 120))
            self.assertTrue(writer.isOpened())
            rng = np.random.default_rng(74)
            frames = [rng.integers(0, 256, (120, 160, 3), dtype=np.uint8) for _ in range(5)]
            for frame in frames:
                writer.write(frame)
            writer.release()
            prepared = prepare_video(source, (160, 120), (80, 60), root/'cache')
            cap = cv2.VideoCapture(str(prepared))
            try:
                self.assertAlmostEqual(cap.get(cv2.CAP_PROP_FPS), 30., places=4)
                for frame in frames:
                    ok, actual = cap.read()
                    self.assertTrue(ok)
                    expected = cv2.resize(frame, (80, 60), interpolation=cv2.INTER_AREA)
                    np.testing.assert_array_equal(actual, expected)
                self.assertFalse(cap.read()[0])
            finally:
                cap.release()
            modified = prepared.stat().st_mtime_ns
            self.assertEqual(prepare_video(source, (160, 120), (80, 60), root/'cache'), prepared)
            self.assertEqual(prepared.stat().st_mtime_ns, modified)
            manifest = json.loads(prepared.with_suffix('.json').read_text())
            self.assertEqual(manifest['frames'], 5)
            self.assertEqual(manifest['identity']['working_size'], [80, 60])
            with self.assertRaisesRegex(ValueError, 'Calibration size'):
                prepare_video(source, (320, 240), (80, 60), root/'wrong')
            self.assertFalse(list((root/'wrong').glob('*.json')))
