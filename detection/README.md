# Day 1 landing-pad detection

The detector finds a visible light quadrilateral containing a large dark X. It
returns four corners and a heuristic quality score. It uses classical OpenCV
operations and does not perform tracking or pose estimation.

## How the files connect

1. `slow_detector.py` reduces the processing width to at most 1280 pixels, converts
   to grayscale and applies Gaussian blur.
2. Canny edges and a complementary Otsu foreground mask produce contours.
   Color inputs also get a near-neutral paper mask (OpenCV HSV saturation < 10,
   value > 100, followed by a 3x3 morphological close). This adds candidates when
   paper and colored fabric have similar brightness; every candidate still has
   to pass the same quadrilateral and X checks. Grayscale inputs use the first
   two masks. This is fixed color thresholding, with no learned model.
3. `approxPolyDP` approximates each contour. Candidate polygons must have four
   convex vertices, sufficient area and edge lengths, and visible image margins.
4. `geometry/homography.py` orders distinct corners clockwise and uses
   `getPerspectiveTransform` / `warpPerspective` to rectify each candidate.
5. `x_detector.py` normalizes the rectified image to a square, extracts dark pixels
   and uses `HoughLinesP` to find both diagonal directions. Nearby dark pixels refine
   each stroke's centerline with `fitLine`. It checks all four arms relative to the
   measured intersection and the fraction of foreground near the fitted lines.
   The crossing can be moderately off-center and strokes can be thin/hand-drawn;
   the detector does not require an exact symmetric diagonal template.
   Otsu thresholding is tried first. If verification fails, Gaussian adaptive
   thresholding (31-pixel neighborhood, offset 10 on the normalized 200-pixel
   image) offers a second mask for unequal stroke contrast. Both masks use the
   same line, arm, intersection and foreground-explanation checks. Candidates
   with more than 43% globally dark foreground remain rejected to avoid accepting
   white-on-black inverse markers. Diagnostics record both attempts.
6. The best verified candidate is returned, with its corners scaled back to the
   original input resolution. If none is verified, the output is invalid.

Lens undistortion and planar perspective rectification are different operations.
The demo can undistort using your camera calibration before running the detector.
The detector itself does not load calibration or apply undistortion a second time.

## Run from the project root

With the virtual environment active, test the generated synthetic example:

```powershell
python -m detection.demo --image outputs/detection_examples/pad_perspective.png
```

For your own still image (use calibration only if it matches that image's capture
mode, not merely the same phone):

```powershell
python -m detection.demo --image path/to/landing_pad.png
```

For a landing-pad video taken with the calibrated 3840 x 2160 capture settings:

```powershell
python -m detection.demo --video videos/landing_pad.mp4 --camera-params calibration/camera_params.npz --sample-every 30 --max-frames 30
```

Supply the actual video file; `landing_pad.mp4` is an example filename. The program
does not open a GUI. Outputs go to `outputs/detection/<input-stem>/`:

- `frame_<id>_annotated.jpg`: the original-size frame with the detected boundary,
  numbered corners and quality score, or an INVALID label.
- `results.json`: validity, original-resolution corners, score, timing, frame ID
  and coordinate-space metadata for every sampled frame.
- `*_gray.png`, `*_edges.png`, `*_paper_mask.png`: intermediate processing images.
- `*_rectified.png`: the winning rectified candidate, when detection succeeds.
- `*_candidate_<id>.png`: rectified candidates, including rejected candidates.
  On debug samples, `results.json` includes `candidate_diagnostics` with the X
  measurements and rejection reason for each candidate. This makes an INVALID
  result inspectable even when the paper boundary was found correctly.

For images, debug intermediates are always saved. For video they are saved for
the first sample; add `--debug` to save them for every sample. `--max-frames` limits
the number of samples processed, not the input frame index. Repeated runs in the
same output directory replace matching filenames; older additional files may
remain, so use a new `--output` directory for a separate experiment.

The video runner samples frames and invokes the full detector on those samples.
This is an offline evaluation tool. The later runtime must still use separate
slow detection and fast per-frame tracking processes.

## Python interface and coordinates

```python
import cv2
from detection.slow_detector import SlowDetector

frame = cv2.imread("path/to/landing_pad.png")
detector = SlowDetector()
result = detector.detect(frame, debug=True)
if result.valid:
    print(result.corners)       # float32, shape (4, 2), input-image pixel coordinates
    print(result.confidence)    # heuristic quality score, not a probability
```

The `DetectionOutput(valid, corners, confidence)` interface remains compatible
with `processes/slow_process.py`. Failed detections return `(False, None, 0.0)`.

The demo's optional calibration requires an exact saved image-size match and uses
the original K without cropping. Corners then refer to the undistorted image, as
recorded in the JSON. Use this same coordinate convention in downstream geometry;
do not undistort these points again. A future uniform image resize must scale K.

Corners are clockwise from the minimum x+y vertex (ties resolved by y, then x).
For a rotated pad this is an image convention, not a physical label on the sheet.
The symmetric X has orientation ambiguity. Physical corner assignment for metric
A4 pose, including the 180-degree ambiguity, must be handled in the pose stage.

## Validation and current limits

Tests cover six rotations, perspective, a small target, dim uneven illumination,
original-resolution coordinates after downsampling, a larger blank distractor,
and rejection of a plus, slash, checkerboard, text and border-only target. Geometry
tests check a diamond in all 24 input permutations and reject degenerate polygons.
Regression tests also cover thin, curved hand-drawn strokes with an off-center
intersection, plus rejection of a three-arm marker and an undersized X.
An additional regression checks paper on a colored background with similar
grayscale brightness, including corner accuracy and negative marker examples.

The checkerboard images test negative detections. The provided landing-pad still
and videos also exercise real positive detections, including colored bedding,
thin hand-drawn strokes and changing distance. These development recordings
are not an independent accuracy benchmark; separate recordings are needed to
assess generalization and false positives when the pad is absent.

This initial detector assumes all four paper corners are visible, usable contrast
with the background, and an X covering much of the sheet. Occlusion, severe motion
blur, very small targets, or a white sheet on a white background may fail. Candidate
work is capped at 100 quads per image; minimum area/size limits are configurable
through the `SlowDetector` constructor. These are processing heuristics, not physical
A4 geometry tests.

No apparent A4 width/height ratio is imposed on the perspective-distorted image.
Conversely, mapping an arbitrary quad to an A4-shaped rectangle does not prove it
is physically A4. A similarly marked rectangle can pass this detector.

```powershell
python -m unittest discover -s tests -v
```
