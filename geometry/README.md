# Day 2, part B: A4 pose

`pose.py` implements classical calibrated pose estimation. No learned model is
used. Four tracked paper corners and a 210 x 297 mm planar A4 model determine a
candidate position and orientation relative to the camera.

## Run

From the project root with the virtual environment active:

```powershell
python -m tracking.demo --video videos/landing_pad2.mp4 --camera-params calibration/camera_params.npz --pose --output outputs/pose_check
```

Add `--max-frames 180` for a short run. The output folder is
`outputs/pose_check/landing_pad2/`; JPG previews contain the results and
`results.json` contains every processed frame. Without `--pose`, the existing
tracking exercise still works. Pose requires calibration matching the lens,
zoom, video mode and pixel coordinates; matching resolution alone is insufficient.

## Follow the calculation

1. Four 3D A4 corners surround the sheet centre at `(0, 0, 0)`. Their coordinates
   are in metres. Object x follows the 210 mm edge, y the 297 mm edge, and z is
   perpendicular to the sheet.
2. Image top-left is not a physical corner label. Try both possible short-edge
   assignments. OpenCV `solvePnPGeneric(..., SOLVEPNP_IPPE)` supplies planar pose
   candidates; `solvePnP(..., SOLVEPNP_ITERATIVE)` refines them. An additional
   iterative seed handles singular IPPE configurations. IPPE_SQUARE is not used:
   the target is a rectangle.
3. Reject nonfinite solutions and any solution with a corner behind the camera.
   `projectPoints` projects the model corners back to pixels. The RMS is the root
   mean square of the four Euclidean pixel distances, in original-resolution
   coordinates. Reject RMS above `AppConfig.max_reprojection_error_px` (5 px).
4. Choose the best fit. With a previous valid pose, prefer rotational continuity
   among fits within 0.25 px of the best and below the acceptance threshold.
   Tracking loss, redetection and rejected pose reset that prior. There is no
   averaging of positions or reuse of stale coordinates.
5. `Rodrigues` converts the rotation vector to a matrix. Euler angles follow
   `R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`, mapping object coordinates into camera
   coordinates. Near gimbal lock the representation uses yaw=0.

## Read the result

- `position_xyz_m`: centre of the sheet in camera coordinates: x right,
  y down, z forward, in metres. This is not the camera's world position.
- `distance_m`: Euclidean camera-to-centre distance `norm(tvec)`. The z component
  is optical-axis depth. Neither value is automatically height above the ground.
- `orientation_rpy_rad` / `orientation_rpy_deg`: roll, pitch, yaw in the stated
  convention. These are the selected sheet axes relative to the camera, not
  compass heading or a uniquely labelled drone orientation.
- `reprojection_error_px`: geometric fit error; low error alone does not prove
  the calibration or physical dimensions are correct.
- `corner_indices`: permutation pairing input image corners with the A4 model.
- `yaw_ambiguous_180`: true because a symmetric X cannot label opposite corners.
  Rotating the object axes by 180 degrees about its normal gives the same image.
  The first pose chooses a representative with yaw nearest zero. Continuity can
  maintain that representative during tracking but cannot resolve the symmetry.
- `orientation_ambiguous`: an additional fitted orientation differs by more than
  5 degrees, even after allowing the 180-degree symmetry, and has RMS within
  0.25 px of the best. This can occur with weak perspective / a small distant pad.

The Python `PoseOutput.orientation_rpy` is in radians. Invalid outputs retain a
reason and, if available, rejected RMS, but position, distance and angles are
`None`. `pose.valid` is separate from the demo's tracking `valid` field.

The demo undistorts frames using the original camera matrix K, then passes K and
zero distortion to PnP. Applying the saved distortion coefficients a second time
would be incorrect. The calibration checkerboard's arbitrary square unit does
not prevent metric A4 pose: the scale here comes from the known A4 dimensions.

## Validation and current limitation

```powershell
python -m unittest discover -s tests -p test_pose.py -v
python tests/validate_pose_demo.py
```

Twelve pose tests cover known metric poses, all cyclic corner orders and reverse
winding, rotations, fronto-parallel views, noisy corners, distorted/undistorted
coordinates, continuity across 90 degrees, ambiguity, invalid inputs, rejection,
Euler conventions and JSON units, calibrated candidate rejection, ROI offsets,
and bright tabletop veins touching a sheet. The 14 detection and 11 tracking
tests also pass.

The reproducible integration script generates a 45-frame video with ground-truth
poses, disappearance and recovery. It uses separate synthetic camera parameters
under `outputs/pose_synthetic/` and never changes the real calibration. The run
accepted 35 poses, cleared output during all eight absent frames, and recovered
after two additional search frames. With the refined detector, maximum position
error was 3.20 mm and median error 2.32 mm on these generated images. These are
synthetic results only.

The initial real 180-frame integration run tracked all 180 frames but rejected every pose.
An additional pose-only audit used all 659 valid tracked corner sets from the
existing 794-frame `landing_pad2` development report: every fit exceeded 5 px,
with best RMS ranging from 6.95 to 17.55 px (median 11.21 px). This audit reused
saved corners; it did not rerun the whole video tracker. Reports are under
`outputs/pose_check/landing_pad2/`.

These were baseline results, before the candidate improvements below. Independent
metric accuracy on real recordings is still not validated. The residual alone cannot distinguish inaccurate corners,
curved paper, non-A4 dimensions or mismatched camera calibration. A useful next
capture is a flat A4 sheet on a rigid surface with the same lens, zoom and capture
mode as calibration. Confirm dimensions and compare estimated distance against
an independent measurement when available. Do not raise the acceptance threshold
just to hide the current mismatch.

## Follow-up: reject wrong boundaries before initializing pose tracking

The calibrated tracking demo now supplies K to the slow detector. Image evidence
first proposes and verifies corners: morphological opening separates thin bright
background veins from the paper; robust straight-edge fits refine contour corners;
and each rectified margin is compared with its neighbouring interior to detect
background intruding into the proposed sheet. Gradual illumination changes remain
allowed. A valid X remains required. The contrast estimate now uses the darkest
0.25% rather than 1% of pixels, because resizing can make thin pen strokes too
sparse for the previous percentile; the line, intersection and four-arm checks
are unchanged.

Each surviving candidate is evaluated with the existing A4 PnP model. Those above
5 px RMS are rejected; the remaining candidates are ranked primarily by their
geometric error. This is a consistency test, not proof of physical accuracy. ROI
searches add their full-image offset only for PnP and still return ROI-local
corners to the recovery wrapper. Uncalibrated detection remains available.
`no_pose_consistent_detection` means an X was found but no candidate passed the
metric geometry gate. It should not be interpreted as a PnP exception.

Full development-video comparison, with the same calibration and 5 px gate:

| Video | Baseline accepted poses | New accepted poses | Frames |
|---|---:|---:|---:|
| Flat table, `landing_pad_table` | 105 | 211 | 353 |
| Bed, `landing_pad2` | 0 | 80 | 794 |

Table tracking losses decreased from 7 to 4. Median RMS over accepted table poses
is 1.70 px (previously 1.39 px on a smaller accepted subset); these populations
differ, so increased availability is not evidence of increased metric accuracy.
On the bed, 80 poses pass during a limited interval, while most views still fail.
The earlier diagnosis remains supported: corner errors contributed, and the
visible curved sheet violates the planar rectangle assumption. The new positive
table fits also weaken the hypothesis that the saved calibration alone explains
all failures; they do not independently validate focal length or distance.

Reviewed overlays and reports: `outputs/table_pose_final/landing_pad_table/`,
`outputs/bed_pose_final/landing_pad2/`, and `outputs/pose_comparison/`.
All 37 targeted tests and the synthetic video integration passed; all 47 real
checkerboard images remained negative for X. Both real recordings are development
data used to diagnose this change, not a held-out accuracy benchmark.

The subsequent bounded local-retry update in `tracking/README.md` raises table
availability to 249/353 using the same pose calculation and error threshold.
See `outputs/recovery_comparison/` for the newer report and search-cost tradeoff.
