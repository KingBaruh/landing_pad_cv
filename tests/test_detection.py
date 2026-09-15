import itertools
import unittest

import cv2
import numpy as np

from detection.slow_detector import SlowDetector
from detection.x_detector import detect_x
from geometry.homography import order_corners, rectify_pad


def make_pad(marker='x'):
    paper = np.full((297, 210, 3), 245, dtype=np.uint8)
    a, b, c, d = (25, 36), (184, 36), (184, 260), (25, 260)
    if marker in ('x', 'slash'):
        cv2.line(paper, a, c, (15, 15, 15), 15, cv2.LINE_AA)
    if marker == 'x':
        cv2.line(paper, b, d, (15, 15, 15), 15, cv2.LINE_AA)
    if marker == 'plus':
        cv2.line(paper, (105, 36), (105, 260), (15, 15, 15), 15)
        cv2.line(paper, (25, 148), (184, 148), (15, 15, 15), 15)
    if marker == 'checkerboard':
        for y in range(7):
            for x in range(10):
                if (x+y) % 2 == 0:
                    cv2.rectangle(paper, (15+x*18, 40+y*30),
                                  (15+(x+1)*18, 40+(y+1)*30), (15, 15, 15), -1)
    if marker == 'text':
        for y in range(50, 260, 40):
            cv2.putText(paper, 'HELLO', (20, y), cv2.FONT_HERSHEY_SIMPLEX, .8, (15,15,15), 2)
    if marker == 'border':
        cv2.rectangle(paper, (15, 15), (194, 281), (15, 15, 15), 12)
    if marker == 'handdrawn_x':
        # Thin, slightly curved strokes meeting right of the geometric center.
        strokes = [np.int32([[30,50],[70,89],[95,115],[126,150],[165,210],[194,255]]),
                   np.int32([[25,257],[83,194],[126,150],[164,102],[184,72]])]
        for stroke in strokes:
            cv2.polylines(paper, [stroke], False, (35,35,35), 3, cv2.LINE_AA)
    if marker == 'three_arms':
        cv2.line(paper, a, c, (15,15,15), 10)
        cv2.line(paper, b, (105,148), (15,15,15), 10)
    if marker == 'tiny_x':
        cv2.line(paper, (80,115), (130,181), (15,15,15), 5)
        cv2.line(paper, (130,115), (80,181), (15,15,15), 5)
    return paper


def render_scene(quad, marker='x', canvas=(960, 720), dim=False):
    width, height = canvas
    rng = np.random.default_rng(2)
    # A textured background and a distracting rectangle, rather than a blank image.
    noise = rng.normal(0, 3, (height, width))
    background = np.clip(65+np.linspace(0, 25, width)[None, :]+noise, 0, 255).astype(np.uint8)
    scene = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(scene, (10, 10), (width-15, height-15), (120, 110, 90), 3)
    paper = make_pad(marker)
    source = np.float32([[0, 0], [209, 0], [209, 296], [0, 296]])
    matrix = cv2.getPerspectiveTransform(source, np.float32(quad))
    warped = cv2.warpPerspective(paper, matrix, canvas)
    mask = cv2.warpPerspective(np.full(paper.shape[:2], 255, np.uint8), matrix, canvas)
    scene[mask > 127] = warped[mask > 127]
    if dim:
        illumination = np.linspace(.38, .8, width)[None, :, None]
        scene = np.clip(scene*illumination, 0, 255).astype(np.uint8)
    return scene


def rotated_quad(angle):
    points = np.float32([[-105,-148.5],[105,-148.5],[105,148.5],[-105,148.5]])
    theta = np.radians(angle)
    rotation = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    return (points @ rotation.T + [480,360]).astype(np.float32)


def corner_error(actual, expected):
    expected = order_corners(expected)
    return min(float(np.linalg.norm(actual-np.roll(expected, shift, axis=0), axis=1).max())
               for shift in range(4))


class GeometryTests(unittest.TestCase):
    def test_diamond_order_is_unique_and_permutation_invariant(self):
        points = np.float32([[0, 50], [50, 0], [100, 50], [50, 100]])
        expected = order_corners(points)
        for permutation in itertools.permutations(points):
            np.testing.assert_array_equal(order_corners(permutation), expected)
        self.assertEqual(len(np.unique(expected, axis=0)), 4)
        self.assertGreater(cv2.contourArea(expected, oriented=True), 0)

    def test_rejects_degenerate_geometry(self):
        for points in ([[0,0]]*4, [[0,0],[1,0],[2,0],[3,0]],
                       [[0,0],[10,0],[10,10],[8,2]]):
            with self.assertRaises(ValueError):
                order_corners(points)

    def test_rectification_recovers_marker_from_perspective(self):
        quad = np.float32([[190,110],[700,170],[610,590],[270,530]])
        rectified = rectify_pad(render_scene(quad), quad[[2,0,3,1]], (210,297))
        self.assertEqual(rectified.shape, (297,210,3))
        self.assertTrue(detect_x(rectified)[0])


class XTests(unittest.TestCase):
    def test_faint_single_pixel_strokes_survive_resize_contrast_check(self):
        for complete in (False, True):
            paper = np.full((396,280),245,np.uint8)
            cv2.line(paper,(30,40),(250,356),150,1,cv2.LINE_AA)
            if complete:
                cv2.line(paper,(250,40),(30,356),150,1,cv2.LINE_AA)
            self.assertEqual(detect_x(paper)[0], complete)

    def test_strokes_with_unequal_contrast_and_incomplete_markers(self):
        for marker in ('x', 'slash', 'three_arms'):
            paper = make_pad('blank')
            cv2.line(paper, (25,36), (184,260), (25,25,25), 3, cv2.LINE_AA)
            if marker == 'x':
                cv2.line(paper, (184,36), (25,260), (180,180,180), 3, cv2.LINE_AA)
            elif marker == 'three_arms':
                cv2.line(paper, (184,36), (105,148), (180,180,180), 3, cv2.LINE_AA)
            for rotation in range(4):
                with self.subTest(marker=marker, rotation=rotation):
                    details = {}
                    valid, _ = detect_x(np.rot90(paper, rotation).copy(), diagnostics=details)
                    self.assertEqual(valid, marker == 'x')
                    if valid:
                        self.assertEqual(details['threshold_method'], 'adaptive_gaussian')
                        self.assertFalse(details['attempts'][0]['valid'])

    def test_thin_handdrawn_x_with_off_center_intersection(self):
        for rotation in range(4):
            with self.subTest(rotation=rotation):
                image = np.rot90(make_pad('handdrawn_x'), rotation).copy()
                self.assertTrue(detect_x(image)[0])

    def test_rotated_markers(self):
        for rotations in range(4):
            with self.subTest(rotations=rotations):
                valid, confidence = detect_x(np.rot90(make_pad(), rotations).copy())
                self.assertTrue(valid)
                self.assertGreater(confidence, .5)

    def test_rejects_distractors(self):
        for marker in ('blank','plus','slash','checkerboard','text','border','three_arms','tiny_x'):
            with self.subTest(marker=marker):
                self.assertFalse(detect_x(make_pad(marker))[0])
        self.assertFalse(detect_x(255-make_pad())[0])


class SlowDetectorTests(unittest.TestCase):
    def test_paper_on_colored_background_with_similar_brightness(self):
        quad = rotated_quad(20)
        mask = np.zeros((720, 960), np.uint8)
        cv2.fillConvexPoly(mask, np.int32(quad), 255)
        for marker in ('x', 'handdrawn_x', 'blank', 'plus', 'slash', 'text', 'tiny_x'):
            with self.subTest(marker=marker):
                scene = render_scene(quad, marker)
                # Very similar grayscale brightness, but different saturation.
                scene[mask == 0] = (180, 250, 245)
                result = SlowDetector().detect(scene)
                self.assertEqual(result.valid, marker in ('x', 'handdrawn_x'))
                if result.valid:
                    self.assertLess(corner_error(result.corners, quad), 6)

    def test_handdrawn_x_on_perspective_sheet(self):
        quad = np.float32([[260,130],[630,220],[690,480],[190,580]])
        result = SlowDetector().detect(render_scene(quad, 'handdrawn_x'))
        self.assertTrue(result.valid)
        self.assertLess(corner_error(result.corners, quad), 6)

    def test_rotation_perspective_small_target_and_illumination(self):
        scenes = [('rotate_'+str(a), rotated_quad(a), False) for a in (0,35,90,135,220,280)]
        scenes += [
            ('perspective', np.float32([[260,130],[630,220],[690,480],[190,580]]), False),
            ('small', np.float32([[710,80],[773,87],[765,174],[702,160]]), False),
            ('dim', rotated_quad(20), True),
        ]
        detector = SlowDetector()
        for name, quad, dim in scenes:
            with self.subTest(name=name):
                output = detector.detect(render_scene(quad, dim=dim))
                self.assertTrue(output.valid)
                self.assertLess(corner_error(output.corners, quad), 6)
                self.assertEqual(output.corners.dtype, np.float32)

    def test_returned_corners_use_original_resolution(self):
        quad = rotated_quad(25)*3
        output = SlowDetector(max_width=960).detect(render_scene(quad, canvas=(2880,2160)))
        self.assertTrue(output.valid)
        self.assertLess(corner_error(output.corners, quad), 14)

    def test_larger_blank_decoy_does_not_hide_target(self):
        quad = np.float32([[660,170],[815,200],[800,420],[640,390]])
        scene = render_scene(quad)
        cv2.rectangle(scene, (60,120), (480,620), (245,245,245), -1)
        output = SlowDetector().detect(scene)
        self.assertTrue(output.valid)
        self.assertLess(corner_error(output.corners, quad), 6)

    def test_negative_scenes_and_reset(self):
        detector = SlowDetector()
        self.assertTrue(detector.detect(render_scene(rotated_quad(0)), debug=True).valid)
        for marker in ('blank','plus','slash','checkerboard','text','border','three_arms','tiny_x'):
            with self.subTest(marker=marker):
                output = detector.detect(render_scene(rotated_quad(20), marker))
                self.assertFalse(output.valid)
                self.assertIsNone(output.corners)
                self.assertEqual(output.confidence, 0.)
        self.assertEqual(detector.debug_images, {})
        for image in (None, np.empty((0,0),np.uint8), np.zeros((40,40),np.float32)):
            self.assertFalse(detector.detect(image).valid)


if __name__ == '__main__':
    unittest.main()
