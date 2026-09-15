# חלק 2 - מעקב באמצעות ראייה ממוחשבת קלאסית

`tracker.py` implements Harris feature selection, pyramidal Lucas-Kanade optical
flow, forward/backward match checking and RANSAC homography estimation. No learned
models are used. Metric A4 pose is available with `--pose`; the three-process
runtime is covered in חלק 3, in `../docs/multiprocessing.md`.

## Follow the code

1. `initialize(frame, corners)` receives a successful detection from that same
   frame. The four observed paper corners are the first four LK anchors. A
   polygon mask limits additional `goodFeaturesToTrack` points to the pad.
   Harris mode favors corners over ambiguous points on straight strokes, and
   points within five working pixels of an anchor are excluded as duplicates.
   Initialization requires six distinct points: all four anchors and at least
   two additional features. Featureless images cannot initialize.
2. `update(next_frame)` uses `calcOpticalFlowPyrLK` to locate the points in the
   next image and then track them backward. Bad status, large intensity error,
   and forward/backward discrepancy reject unreliable correspondences. All four
   anchors must survive these checks; losing a corner triggers redetection.
3. `estimate_homography` fits the plane transformation using `findHomography`
   with RANSAC. Correspondences retain coordinates from initialization, avoiding
   repeated multiplication of noisy incremental homographies. `transform_corners`
   applies that transformation and preserves vertex identities across rotation.
   If RANSAC extrapolates more than 1.5 working pixels away from the tracked
   corners, their four correspondences define an alternative homography using
   `getPerspectiveTransform`. Additional features must independently agree with
   the selected model within 2.5 working pixels. Anchors remain first in the
   point arrays across filtering and replenishment.
4. Checks require six inliers, including all four anchors,
   at least 50% agreement, spatial coverage in both
   the previous image and the initialization reference (at least 20% pad area), a
   convex visible quadrilateral and bounded changes in area and motion. The
   numerical thresholds use the working image, not the full-resolution image.
5. Every ten updates, the tracked crop is checked using the existing classical
   X verifier. This checks the proposed location without searching the whole
   image. Failed verification triggers loss; it does not guarantee zero drift
   between checks. Strong features may then be replenished inside the
   pad. Their reference coordinates come from the inverse current homography.
   Replenished points stay away from the boundary to avoid acquiring fabric.
   Feature quality is always measured against the whole pad, so excluding
   existing points does not accidentally promote weak edge points.
6. A failed update clears the old corners and points and reports a reason.
   The caller must detect again before restarting tracking.

The tracker downsizes grayscale images to at most 1280 pixels wide. Public corner
coordinates are always in the original input resolution. If the caller undistorts
frames, corners are in that undistorted coordinate system. Do not undistort twice.
The initial image corner order is not a unique physical orientation of symmetric X.

## Run the continuous-video exercise

From the project root with the virtual environment active:

```powershell
python -m tracking.demo --video videos/landing_pad2.mp4 --camera-params calibration/camera_params.npz
```

The demo processes **every frame**, including the frames between saved previews.
`--save-every 30` (default) saves one preview per 30 frames, plus initialization
and loss events. `--max-frames 150` limits processing for a short exercise; the
normal default processes the whole video. Calibration must match the capture
mode and dimensions; omit it for an unmatched input.

For pose, calibration is required:

```powershell
python -m tracking.demo --video videos/landing_pad2.mp4 --camera-params calibration/camera_params.npz --pose --output outputs/pose_check
```

This adds camera-relative position, distance, roll/pitch/yaw, projected axes and
reprojection RMS. A per-frame `pose` object records validity separately from
tracking validity. A rejected pose has no position or angles; its reason and
RMS remain available. Missing tracking corners clear pose continuity immediately,
including loss followed by redetection on the same frame. `pose_ms` measures the
pose calculation and `pose_valid_frames` counts accepted metric estimates.
Frames are already undistorted with the original K, so PnP receives zero distortion.
See `geometry/README.md` for units, the X symmetry and current real-video results.
With `--pose`, candidate A4 reprojection checks also run during detection before
LK initialization. ROI coordinates are translated to the calibrated full frame
for that check. `no_pose_consistent_detection` distinguishes geometric rejection
from an ordinary failure to find the marker. Detection without `--pose` remains
available for unmatched camera modes and for inspecting nonplanar targets.

Outputs are in `outputs/tracking/landing_pad2/`:

- Preview JPGs: green paper boundary and red tracked points; previews are resized
  to at most 1280 pixels wide for convenient viewing.
- `results.json`: original-resolution corners, per-frame state/reason, inlier
  counts/ratio, LK error, timings and the source (`detector` or `tracker`).
  X-check entries also retain proposed corners, threshold method and rejection
  measurements even if the tracker resets or reinitializes on the same frame.
  With `--debug`, separate X crop and candidate JPG/PNG images are saved for each
  check. Amber candidate outlines are diagnostic proposals, not valid results.

Detection is attempted immediately on the first frame and then only while the
tracker is uninitialized. `--redetect-every 30` limits searches that may fall
back to the full frame. Between those deadlines, `--local-retry-every 5` allows
an early local retry for a recently lost pad. These retries expire one full-search
interval after the last valid tracked frame. A lost track can be reinitialized
on the same frame if an attempt is due, subject to the local cooldown.
`recovery.py` first searches a padded rectangle around the last tracked position.
For scheduled full searches, a local failure falls back to the whole frame.
Early retries use a tighter window (25% margin rather than 65%) and never invoke
the full-frame detector. Local corners are
translated back to full-frame pixels. This can separate the paper from clutter
that confused global thresholding. `detection_scope` records the chosen route;
one scheduled recovery attempt can make two detector calls, counted in
`detector_calls`. `recovery_mode` records `full` (fallback permitted) or `local`;
it differs from `detection_scope`, which records the actual successful/final
search. `full_search_attempts` and `local_retry_attempts` count actual attempts;
an unavailable ROI with zero detector calls is not counted.
The JSON records that loss separately even when recovery is immediate. Newly
initialized frames are not also sent through LK a second time.

This demo runs synchronously: it never applies an old detection to a new frame.
It is not `main.py` or the final multiprocessing application. Valid-frame counts
measure reported availability, not independently annotated accuracy; visual
checks and separate held-out videos remain necessary. Thin strokes, too few
corners, occlusion, motion blur and long-term drift can still cause failures.

## Tests

```powershell
python -m unittest discover -s tests -p test_tracking.py -v
python -m unittest discover -s tests -p test_detection.py -v
```

Tracking tests use known motion to check corner accuracy, moving paper over a
stationary background, original-resolution coordinates after resizing, incorrect
point matches, degeneracy, disappearance, reset and reinitialization.

Real-video validation on all 794 frames of `landing_pad2.mp4`, with calibration:

| Measure | Previous baseline | Corner anchors + local recovery |
|---|---:|---:|
| Reported valid frames | 190 (23.9%) | 502 (63.2%) |
| Tracking losses | 9 | 7 |
| Recovery attempts | 25 | 17 |
| Actual detector calls (local + global) | 25 | 28 |

Reports, previews and a before/after contact sheet are in
`outputs/tracking_improved/landing_pad2/`. In reviewed sampled frames the tracked
boundaries follow the paper. Overlap with previously reviewed detector outputs
is recorded in `comparison.json`; these are development checks, not independent
ground truth. Some searches still fail and marker checks can interrupt tracking.
The earlier attempt merely lowering the minimum point count drifted; this version
additionally requires explicit corner anchors and supporting interior matches.
Validate on separate recordings before relying on tracked corners for metric pose.

## Follow-up: unequal-contrast X strokes

Audit of frames 346, 576, 646 and 676 found the outline still on the paper, with
an X visible in the crop. Otsu lost part of a lighter stroke or made the line
evidence insufficient for Hough. The X verifier now tries local Gaussian
thresholding after Otsu fails, retaining all existing geometric checks. All four
original crops pass without changing their corners. Synthetic regressions cover
unequal-contrast diagonals in four rotations and rejection of incomplete markers.

On the same 794-frame video, reported valid frames increased from 502 to 659
(83.0%). Total tracking losses changed from 7 to 10: the longer tracked sequence
encounters other failures, so the fix does not eliminate all interruptions.
Remaining X rejections occurred at frames 436, 694, 754 and 764; other losses
were due to too few inliers. The 47 checkerboard negatives had zero detections.
The 13 detection tests and 11 tracking tests passed. Full audit details and
before/after threshold images are in `outputs/tracking_x_verified/landing_pad2/`.
These remain development-video results, not an independent accuracy benchmark.

## Follow-up: bounded local recovery retries

The earlier fixed search interval delayed recovery by up to nearly 30 frames
after a brief loss. `RecoverySchedule` now permits a short burst of tighter local
searches between full-search deadlines. It preserves all existing feature, X
and calibrated pose acceptance criteria. For example, after the loss at table
frame 130, the new version recovered at frame 135 instead of frame 160.

On the 353-frame table video, accepted poses increased from 211 to 249 (70.5%).
Actual detector calls increased from 11 to 21, while actual full-frame calls
changed from 7 to 6. Tracking losses changed from 4 to 5 as different portions
of the recording became tracked. Median accepted-pose RMS changed from 1.70 to
1.72 px; these are different accepted populations, not a metric accuracy test.
The tradeoff is more local computation for shorter gaps. This remains an offline
exercise, not a claim of real-time throughput.

Fourteen tracking tests and twelve pose tests passed, including recovery after
short disappearance, no stale pose during absence, expiry of local retries,
preserved full-search cadence, and escape from an incorrect local window at the
next full search. The 45-frame generated video still has 35 accepted poses and
maximum position error 3.21 mm; it now recovers using local retries with only
one full-search opportunity. Results: `outputs/table_local_recovery/` and
`outputs/recovery_comparison/`.

## Follow-up: recovery after sustained Pose rejection

With `--pose`, three consecutive rejected tracked poses now reset the tracker;
the existing recovery scheduler then reacquires from image evidence. Configure
the count with `--pose-reset-after` (default 3). A valid Pose clears the count.
The calibrated Pose threshold remains 5 px; tracking without Pose is unchanged.
JSON includes `pose_triggered_resets`, `pose_rejection_streak`, and diagnostic
`rejected_tracking_corners`. A reset reports `pose_inconsistent` and no current
corners; invalid metric results remain unavailable throughout rejection/search.

Same-frame comparisons showed accumulated corner displacement despite successful
LK/H and X checks: fresh local detection reduced frame 560 RMS from 11.91 to
1.03 px with the same calibration/solver. Full-video accepted Pose counts changed
606 to 645 on the new take and 249 to 276 on the old tabletop; the bed remained
80. Some individual recovery gaps worsened, so this is a targeted feedback fix,
not uniform tracking improvement or proof of metric accuracy. The new take has
now been used for development and requires a further unseen validation recording.

Fourteen tracking tests, twelve Pose tests, a new injected-drift integration
regression, and the known-pose synthetic video passed. Evidence, exact comparison
spans and remaining regressions: `outputs/pose_drift_diagnosis/README.md`.
