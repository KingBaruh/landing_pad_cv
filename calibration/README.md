# Camera calibration — Day 1

This is the separate, offline calibration application required by the assignment.
It estimates the camera matrix and lens distortion from checkerboard images.
Real phone calibration still requires captured images; the automated tests use
synthetic images and do not produce parameters for your phone.

## Capture setup

- Start with the phone's rear main camera at 1x, landscape FHD (1920 x 1080), 30 FPS.
- Keep the same lens, resolution, orientation and capture settings for calibration
  and the landing-pad test videos. Avoid zooming and automatic lens switching.
- Disable Super Steady and electronic video stabilization where the camera app
  permits it. Dynamic cropping/warping can violate the fixed camera model.
- Use good, even lighting and keep the board sharp. Keep focus fixed when practical.
- Extract calibration frames from video recorded in the test capture mode; still
  photos can have different cropping and camera parameters.
- Use a flat printed checkerboard with 10 x 7 squares. A suggested square side is
  25 mm; it fits on landscape A4. Print at 100% and measure the actual side length.
- This board has **9 x 6 inner corners**. The calibration board is a separate
  pattern from the landing pad marked with an X.
- Collect 15–20 sharp, varied views: tilt around both axes, vary distance, and move
  the board across the image. Keep the whole board visible. Pause at each pose.
- Put extracted PNG/JPEG frames into `calibration/images/`. Frame extraction and
  the optional live capture helper are not implemented in this step.

## What the code does

### Alternative without a printer or ruler

Open `calibration/checkerboard_screen.svg` in a browser on a flat computer screen.
Keep all 10 x 7 squares and their white margin visible, preserve the aspect ratio,
and do not resize the pattern during capture. Move the phone around the stationary
screen to obtain different tilts and positions. Avoid reflections, visible flicker,
moire patterns and overexposure; first record a short sample and inspect it.

Use `--square-size 1` for this option. Object coordinates are in square units.
The camera matrix and distortion do not require the absolute square size; board
translations would be in square units, not meters. Later metric landing-pad pose
uses the real A4 dimensions (0.210 x 0.297 m), independently of board scale.
A screen is a practical starting target; quality must be checked on real captures.

### Computation

1. Build known board coordinates: `(0,0,0), (s,0,0), (2s,0,0), ...`, where `s` is
   the measured square side. All points lie on the board plane, so `Z = 0`.
2. Use `findChessboardCorners` to locate the corresponding image pixels on a
   detection image no wider than 1280 pixels.
3. Scale the detected points back to original image coordinates and refine them
   with `cornerSubPix` on the original image. K stays in original-resolution pixels.
4. Fit a shared camera matrix `K` and distortion coefficients using
   `calibrateCamera`. Each view also has its own board rotation and translation.
5. Project board points back into each image and measure RMS pixel error.
6. Save parameters and diagnostic images.

Ignoring lens distortion, the model is `lambda * [u,v,1]^T = K [R|t] [X,Y,Z,1]^T`.
Here `K = [[fx,0,cx], [0,fy,cy], [0,0,1]]`; focal lengths and the principal point
are in pixels. Lens distortion is fitted in addition to this projection.

Per-view RMS is `sqrt(mean((u_observed-u_predicted)^2 + (v_observed-v_predicted)^2))`.
A low fitting error alone does not guarantee reliable calibration. Inspect corner
overlays, view diversity and an independent image before trusting measurements.

## Run from the project root

After capturing images, for measured 25 mm squares:

```powershell
.venv\Scripts\python.exe -m calibration.calibrate --square-size 0.025
```

For other board dimensions, specify **inner corners**:

```powershell
.venv\Scripts\python.exe -m calibration.calibrate --images calibration/images --board-cols 9 --board-rows 6 --square-size 0.025
```

Outputs:

- `calibration/camera_params.npz`: `camera_matrix`, `dist_coeffs`, image size,
  board dimensions, square size, RMS errors, and used/skipped image paths.
- `calibration/debug/*_corners.png`: inspect the measured corner ordering.
- `calibration/debug/undistortion_before_after.png`: original on the left,
  corrected image on the right. This uses a calibration view, not a held-out one.

At least 10 usable views are required by this implementation. Missing inputs,
insufficient detected boards, and mixed resolutions produce explicit errors.

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The tests recover known intrinsics from 12 rendered board poses, check save/load,
and verify rejection of missing inputs, insufficient views and mixed resolutions.

## Course reading and next step

The second recording has now produced `calibration/camera_params.npz`, using
45 of 47 sampled views at 3840 x 2160. The fitting RMS is 0.5107 pixels. For this
dataset the selected model fits k1, p1 and p2 while fixing k2 and k3 to zero:

```powershell
.venv\Scripts\python.exe -m calibration.calibrate --images calibration/images/chessboard_2 --square-size 1 --radial-order 1 --exclude frame_00660.png frame_00990.png --output calibration/camera_params.npz --debug-dir outputs/calibration/chessboard_2/selected_debug
```

The two excluded images were visually reviewed as blurred/deformed. The originals
remain available. See `outputs/calibration/chessboard_2/report.md` for selection,
model comparisons, validation and remaining limitations. This is a development
calibration; metric accuracy at frame edges still needs independent validation.

Sampling can be reproduced from the project root with:

```powershell
.venv\Scripts\python.exe tmp/chessboard_review/analyze_video.py --video chessboard_2 --step 30
```

The sampling and diagnostic scripts are currently task analysis helpers in `tmp/`.

- `חוברת.pdf`, PDF pages 29–30: camera matrix, intrinsics and extrinsics.
- `חוברת.pdf`, PDF page 34: distortion and calibration with a planar board.
- `summary_2024.pdf`, PDF pages 93 and 95: Canny and Hough for the later detectors.

After real calibration and inspection, continue with `detection/slow_detector.py`
and then `detection/x_detector.py`. Rectification will also require implementing
the relevant helpers in `geometry/homography.py`.
