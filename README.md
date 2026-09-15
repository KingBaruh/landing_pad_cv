# Landing Pad Detection and Tracking

Classical OpenCV solution for detecting, tracking, and estimating the pose of an A4 landing pad marked with a large black X.

For installation and full runtime instructions, see the [run guide](RUNNING.md).

The consolidated Hebrew [technical report](docs/technical_report.html) includes
architecture, algorithms, synchronization, run instructions, eight experiments,
and embedded performance graphs. Its [editable source](docs/technical_report.md)
and [experiment descriptions](docs/experiment_descriptions.md) include the complete
v1 rerun (309 source frames, 289 processed frames, 253 accepted poses).

## Architecture

- Video Player Process
- Fast Algorithm Process
- Slow Algorithm Process
- Separate Camera Calibration application

## Suggested development order

1. Camera calibration
2. Slow detector
3. X verification
4. Pose estimation
5. Fast tracker
6. Tracking-loss detection and re-detection
7. Split into three processes
8. Metrics, test videos, and documentation

## Part 1 - Camera calibration and landing pad detection

### Run camera calibration

From the project root, with the virtual environment active, install the dependencies if needed:

```text
python -m pip install -r requirements.txt
```

To reproduce the calibration using the existing checkerboard images in `calibration/images` and the settings selected for this project:

```text
python calibration/calibrate.py --square-size 1 --radial-order 1 --exclude frame_00660.png frame_00990.png
```

- `--square-size 1` uses an arbitrary square unit, as used when calibrating with a checkerboard displayed on a screen.
- `--radial-order 1` fits one radial distortion coefficient in addition to the tangential distortion coefficients.
- `--exclude` skips the two images selected for exclusion during calibration review. For a new image set, review which images should be excluded.
- The default checkerboard size is 9 by 6 inner corners, not squares.

The program prints the RMS reprojection error, camera matrix and distortion coefficients. Parameters are saved to `calibration/camera_params.npz`, and diagnostic images are saved under `calibration/debug/`. This command overwrites the existing parameter file; to save a separate file, add `--output calibration/camera_params_recomputed.npz`.

The runtime loads the saved parameters, so calibration does not need to be repeated for every run. See the [calibration documentation](calibration/README.md) for details.

### Run detection

```bash
python -m detection.demo --image outputs/detection_examples/pad_perspective.png
```

The example image is synthetic. See `calibration/README.md` for calibration and
`detection/README.md` for the detector, video commands and output formats.

To process a real landing-pad video with matching calibration:

```bash
python -m detection.demo --video videos/landing_pad.mp4 --camera-params calibration/camera_params.npz --sample-every 30 --max-frames 30
```

Supply your own `videos/landing_pad.mp4`. The existing checkerboard videos are
negative examples for X detection, so `Invalid` is expected on them.

## Part 2 - Tracking and pose estimation

The slow detector, X verification, corner ordering and perspective rectification
are implemented. This part adds a classical Harris/Lucas-Kanade tracker,
RANSAC homography updates and a continuous-video exercise:

```powershell
python -m tracking.demo --video videos/landing_pad2.mp4 --camera-params calibration/camera_params.npz
```

See `tracking/README.md` for the algorithm and output details. The exercise reads
every frame and redetects after loss. Metric A4 pose is now available with
`--pose` and matching `--camera-params`; see `geometry/README.md` for conventions,
validation and limitations.

## Part 3 - Three-process runtime

```powershell
python main.py --video videos/landing_pad_test.mp4
```

This starts separate Video, Fast, and Slow processes using Windows-compatible
`spawn`. Calibration is loaded from `calibration/camera_params.npz`. Press Q in
the video window or Ctrl+C in the terminal to stop. A headless run is available:

```powershell
python main.py --video videos/landing_pad_test.mp4 --headless --output outputs/runtime
```

After the workers finish, `main.py` automatically creates the three performance
graphs and a combined overview in `<output>/graphs/`, in PNG and SVG, with Hebrew
explanations. The acceptance limit is read from that run's `results.json`.
Graceful early stops also export graphs if at least two frames were saved.
Failed/incomplete runs and runs shorter than two frames are explicitly skipped;
`runtime_summary.json` records the export status. Outputs include per-frame results, metrics CSV, sampled annotated
images, and process PID/exit summaries. Video encoding is not required or enabled.

Delayed slow detections are initialized on their actual source frame and tracked
through a bounded history before use. They are never attached directly to a newer
frame. The runtime preserves the calibration's pixel-error gate when resizing.
Image queues are bounded and may drop frames under load. The current measured
real-video throughput is below 30 FPS; multiprocessing alone does not establish
real-time performance or metric distance accuracy.

See [the runtime design and validation](docs/multiprocessing.md) for communication,
coordinate conventions, shutdown, measured results, and remaining limitations.

## Tests

```bash
python -m unittest discover -s tests -v
```
