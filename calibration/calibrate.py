import argparse
from pathlib import Path
from xml.parsers.expat import errors
import cv2
import numpy as np


def calibrate_camera(images_dir: str, board_size=(9, 6), square_size=1.0,
                     *, min_views=10, debug_dir=None, max_detection_width=1280,
                     radial_order=3, exclude_images=()):
    """Return intrinsics and quality measurements from checkerboard images.

    board_size is (inner columns, inner rows), NOT the number of squares.
    square_size is the measured square side; use meters for metric translations.
    All images must use the same camera, lens, resolution and capture settings.
    min_views is a practical guard, not a guarantee of a good calibration.
    This function computes parameters; the CLI below saves them separately.
    Detection can use a smaller image, but refinement and K use original pixels.
    radial_order controls how many radial terms (k1, k2, k3) are fitted.
    exclude_images contains explicitly reviewed, unusable image basenames.
    """
    if (len(board_size) != 2 or
            any(not isinstance(n, (int, np.integer)) or n < 2 for n in board_size)):
        raise ValueError("board_size must contain two integers >= 2.")

    if not np.isfinite(square_size) or square_size <= 0:
        raise ValueError("square_size must be positive and finite.")

    if not isinstance(min_views, int) or min_views < 3:
        raise ValueError("min_views must be an integer >= 3.")
    if not isinstance(max_detection_width, int) or max_detection_width < 100:
        raise ValueError("max_detection_width must be an integer >= 100.")
    if radial_order not in (1, 2, 3):
        raise ValueError("radial_order must be 1, 2 or 3.")

    folder = Path(images_dir)
    if not folder.is_dir():
        raise FileNotFoundError(f"Calibration image directory not found: {folder}")

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    paths = sorted(p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in extensions)

    if not paths:
        raise ValueError(f"No supported calibration images in {folder}.")
    excluded_names = set(exclude_images)
    unknown_names = excluded_names - {p.name for p in paths}
    if unknown_names:
        raise ValueError(f"Excluded images not found: {sorted(unknown_names)}")

    debug_path = Path(debug_dir) if debug_dir is not None else None

    if debug_path is not None:
        if debug_path.resolve() == folder.resolve():
            raise ValueError("debug_dir must differ from the input image directory.")

        debug_path.mkdir(parents=True, exist_ok=True)

    # 1. Known 3D positions on the flat board. Every point has Z = 0.
    columns, rows = board_size
    board_points = np.zeros((columns * rows, 3), dtype=np.float32)
    board_points[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    board_points *= square_size

    object_points, image_points, used_images = [], [], []
    skipped_images = []
    image_size = None
    first_image = None
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # 2. Measure corresponding 2D pixel positions in each view.
    for path in paths:
        if path.name in excluded_names:
            print(f"Excluded by explicit selection: {path.name}")
            skipped_images.append(str(path))
            continue
        frame = cv2.imread(str(path)) # Read the image from the file

        if frame is None:
            print(f"Skipped unreadable image: {path.name}")
            skipped_images.append(str(path))
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_size = (gray.shape[1], gray.shape[0])  # OpenCV: width, height

        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            raise ValueError(f"Mixed image resolutions: {path.name} is {current_size}, "
                             f"expected {image_size}. Use one capture mode.")

        # Large screen patterns were missed by direct 4K detection. Detect at
        # a manageable scale, then convert coordinates back to original pixels.
        detection_gray = gray
        if gray.shape[1] > max_detection_width:
            resized_height = round(gray.shape[0] * max_detection_width / gray.shape[1])
            detection_gray = cv2.resize(gray, (max_detection_width, resized_height),
                                        interpolation=cv2.INTER_AREA)
        found, corners = cv2.findChessboardCorners(
            detection_gray, board_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if not found:
            print(f"Skipped (board not found): {path.name}")
            skipped_images.append(str(path))
            continue

        scale_xy = np.array([gray.shape[1] / detection_gray.shape[1],
                             gray.shape[0] / detection_gray.shape[0]], dtype=np.float32)
        corners = (corners.reshape(-1, 2) * scale_xy).reshape(-1, 1, 2)

        # 3. Refine the measured corners to sub-pixel precision.
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        object_points.append(board_points.copy())
        image_points.append(corners)
        used_images.append(str(path))
        if first_image is None:
            first_image = frame.copy()
        print(f"Board found: {path.name}")
        if debug_path is not None:
            cv2.drawChessboardCorners(frame, board_size, corners, found)
            _write_png(debug_path / f"{path.name}_corners.png", frame)

    if len(used_images) < min_views:
        raise ValueError(f"Only {len(used_images)} usable views; need at least {min_views}. "
                         "Check inner-corner dimensions and collect varied board poses.")

    # 4. Fit one shared camera model and a different board pose for each image.
    flags = 0
    if radial_order < 3:
        flags |= cv2.CALIB_FIX_K3
    if radial_order < 2:
        flags |= cv2.CALIB_FIX_K2
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None, flags=flags,
    )
    if not all(np.isfinite(a).all() for a in (rms, camera_matrix, dist_coeffs)):
        raise ValueError("Calibration produced non-finite parameters. Check the views.")

    # 5. Reproject the board and measure RMS Euclidean pixel error per view.
    # These are fitting errors on the calibration images, not held-out accuracy.
    errors = []
    for obj, measured, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, camera_matrix, dist_coeffs)
        residual = measured.reshape(-1, 2) - projected.reshape(-1, 2)
        errors.append(float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1)))))

    if debug_path is not None:
        corrected = cv2.undistort(first_image, camera_matrix, dist_coeffs)
        _write_png(debug_path / "undistortion_before_after.png",
                   np.hstack((first_image, corrected)))

    return {
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "image_size": np.asarray(image_size),
        "board_size": np.asarray(board_size),
        "square_size": float(square_size),
        "radial_order": radial_order,
        "calibration_flags": flags,
        "max_detection_width": max_detection_width,
        "excluded_images": np.asarray(sorted(excluded_names), dtype=str),
        "rms_px": float(rms),
        "per_view_rms_px": np.asarray(errors),
        "used_images": np.asarray(used_images),
        "skipped_images": np.asarray(skipped_images, dtype=str),
    }


def _write_png(path, frame):
    """Use NumPy file I/O so Windows paths containing Hebrew also work."""
    success, encoded = cv2.imencode(".png", frame)
    if not success:
        raise OSError(f"Could not encode image: {path}")
    encoded.tofile(path)


def save_camera_params(path: str, camera_matrix, dist_coeffs, **metadata):
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("wb") as stream:
        np.savez(stream, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, **metadata)


def load_camera_params(path: str):
    path_obj = Path(path)
    if not path_obj.exists():
        # Keep the skeleton runnable enough to fail clearly.
        return {
            "camera_matrix": None,
            "dist_coeffs": None,
        }

    with np.load(path_obj, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline checkerboard camera calibration")
    parser.add_argument("--images", default="calibration/images")
    parser.add_argument("--board-cols", type=int, default=9, help="Inner corners across")
    parser.add_argument("--board-rows", type=int, default=6, help="Inner corners down")
    parser.add_argument("--square-size", type=float, required=True,
                        help="Square side: 0.025 for 25 mm, or 1 for square units "
                             "when only camera intrinsics/distortion are needed")
    parser.add_argument("--output", default="calibration/camera_params.npz")
    parser.add_argument("--debug-dir", default="calibration/debug")
    parser.add_argument("--max-detection-width", type=int, default=1280)
    parser.add_argument("--radial-order", type=int, choices=(1, 2, 3), default=3,
                        help="Number of fitted radial distortion terms (default: 3)")
    parser.add_argument("--exclude", nargs="*", default=[], metavar="BASENAME",
                        help="Explicitly reviewed unusable images to exclude")
    args = parser.parse_args()

    try:
        result = calibrate_camera(args.images, (args.board_cols, args.board_rows),
                                  args.square_size, debug_dir=args.debug_dir,
                                  max_detection_width=args.max_detection_width,
                                  radial_order=args.radial_order, exclude_images=args.exclude)
        save_camera_params(args.output, **result)
    except (OSError, ValueError, cv2.error) as error:
        parser.exit(1, f"Calibration failed: {error}\n")

    print(f"\nUsed {len(result['used_images'])} images; RMS: {result['rms_px']:.4f} px")
    print("Camera matrix:\n", result["camera_matrix"])
    print("Distortion coefficients:", result["dist_coeffs"].ravel())

    for name, error in zip(result["used_images"], result["per_view_rms_px"]):
        print(f"  {Path(name).name}: {error:.4f} px RMS")

    median_error = np.median(result["per_view_rms_px"])
    max_error = np.max(result["per_view_rms_px"])

    print(f"Median error: {median_error:.4f} px")
    print(f"Max error: {max_error:.4f} px")

    print(f"Saved parameters: {args.output}")
    print(f"Inspect detected corners and before/after images in: {args.debug_dir}")
