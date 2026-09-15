"""Offline Day 1 demonstration: python -m detection.demo --help.

Video sampling here evaluates the slow detector; it is not the three-process
tracking application. Every returned corner uses the original frame resolution.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from calibration.calibrate import load_camera_params
from detection.slow_detector import SlowDetector


def draw_detection(frame, result):
    overlay = frame.copy()
    scale = max(1., frame.shape[1]/1280)
    color = (0, 210, 0) if result.valid else (0, 0, 230)
    if result.valid:
        points = np.round(result.corners).astype(np.int32)
        cv2.polylines(overlay, [points], True, color, round(2*scale), cv2.LINE_AA)
        for i, (x, y) in enumerate(points):
            cv2.circle(overlay, (int(x), int(y)), round(4*scale), (0, 0, 255), -1)
            cv2.putText(overlay, str(i), (int(x+8*scale), int(y-8*scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, .65*scale, color, round(2*scale))
    label = f"{'VALID' if result.valid else 'INVALID'}  quality={result.confidence:.2f}"
    cv2.putText(overlay, label, (round(16*scale), round(32*scale)),
                cv2.FONT_HERSHEY_SIMPLEX, .8*scale, color, round(2*scale), cv2.LINE_AA)
    return overlay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--image', type=Path)
    source.add_argument('--video', type=Path)
    parser.add_argument('--output', type=Path, default=Path('outputs/detection'))
    parser.add_argument('--camera-params', type=Path,
                        help='Optional calibration for this exact capture mode and resolution')
    parser.add_argument('--sample-every', type=int, default=30, help='Video frame sampling interval')
    parser.add_argument('--max-frames', type=int, default=30, help='Maximum video samples to process')
    parser.add_argument('--debug', action='store_true', help='Save intermediate images for every sample')
    args = parser.parse_args()
    if args.sample_every < 1 or args.max_frames < 1:
        parser.error('--sample-every and --max-frames must be positive')
    source_path = args.image or args.video
    if not source_path.is_file():
        parser.error(f'Input file not found: {source_path}')
    params = None
    if args.camera_params is not None:
        params = load_camera_params(str(args.camera_params))
        if params.get('camera_matrix') is None or 'image_size' not in params:
            parser.error('Calibration file must contain camera_matrix, dist_coeffs and image_size')

    output = args.output / source_path.stem
    output.mkdir(parents=True, exist_ok=True)
    detector = SlowDetector()
    rows = []
    maps = None
    capture = None

    def process(frame, frame_id, timestamp):
        nonlocal maps
        height, width = frame.shape[:2]
        start = perf_counter()
        if params is not None:
            if tuple(params['image_size']) != (width, height):
                raise ValueError(f'Calibration size {tuple(params["image_size"])} does not match '
                                 f'input {(width, height)}. Use matching calibration/capture settings.')
            if maps is None:
                k = params['camera_matrix']
                maps = cv2.initUndistortRectifyMap(k, params['dist_coeffs'], None, k,
                                                 (width, height), cv2.CV_32FC1)
            frame = cv2.remap(frame, maps[0], maps[1], cv2.INTER_LINEAR)
        debug = args.debug or args.image is not None or not rows
        result = detector.detect(frame, debug=debug)
        elapsed_ms = (perf_counter()-start)*1000
        prefix = f'frame_{frame_id:06d}'
        path = output / (prefix+'_annotated.jpg')
        if path.resolve() == source_path.resolve():
            raise ValueError('Output would overwrite the input image; choose another output folder.')
        if not cv2.imwrite(str(path), draw_detection(frame, result)):
            raise OSError(f'Cannot write {path}')
        if debug:
            for name, image in detector.debug_images.items():
                debug_path = output / (prefix+'_'+name+'.png')
                if not cv2.imwrite(str(debug_path), image):
                    raise OSError(f'Cannot write {debug_path}')
        row = dict(frame_id=frame_id, timestamp_s=timestamp, valid=result.valid,
                   confidence=result.confidence,
                   corners=result.corners.tolist() if result.valid else None,
                   frame_size=[width, height], candidate_count=detector.last_candidate_count,
                   processing_ms=elapsed_ms, annotated_image=str(path))
        if debug:
            row['candidate_diagnostics'] = detector.candidate_diagnostics
        rows.append(row)
        print(f'{prefix}: valid={result.valid}, quality={result.confidence:.3f}, '
              f'candidates={detector.last_candidate_count}, {elapsed_ms:.1f} ms', flush=True)

    try:
        if args.image is not None:
            frame = cv2.imread(str(args.image))
            if frame is None:
                raise ValueError(f'Cannot read image: {args.image}')
            process(frame, 0, None)
        else:
            capture = cv2.VideoCapture(str(args.video))
            if not capture.isOpened():
                raise ValueError(f'Cannot open video: {args.video}')
            fps = capture.get(cv2.CAP_PROP_FPS)
            index = 0
            while len(rows) < args.max_frames:
                success, frame = capture.read()
                if not success:
                    break
                if index % args.sample_every == 0:
                    process(frame, index, index/fps if fps > 0 else None)
                index += 1
        if not rows:
            raise ValueError('No readable frames in input.')
    except (OSError, ValueError, cv2.error) as error:
        parser.exit(1, f'Detection failed: {error}\n')
    finally:
        if capture is not None:
            capture.release()

    report = dict(source=str(source_path),
                  camera_params=str(args.camera_params) if params is not None else None,
                  coordinate_space='undistorted_original_resolution' if params is not None else 'input_original_resolution',
                  sampled_frames=len(rows), valid_frames=sum(r['valid'] for r in rows), frames=rows)
    path = output / 'results.json'
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Saved {len(rows)} sampled results to {path}')


if __name__ == '__main__':
    main()
