"""Sequential Day 2 tracker exercise: python -m tracking.demo --help.

Processes every frame; --save-every only controls saved preview images.
This is an offline runner, not the future three-process application or pose stage.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from calibration.calibrate import load_camera_params
from detection.slow_detector import SlowDetector
from tracking.tracker import LandingPadTracker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--camera-params', type=Path)
    parser.add_argument('--output', type=Path, default=Path('outputs/tracking'))
    parser.add_argument('--max-frames', type=int, default=0, help='0 processes the whole video')
    parser.add_argument('--save-every', type=int, default=30, help='Save a preview every N frames')
    parser.add_argument('--redetect-every', type=int, default=30,
                        help='Minimum interval between detection attempts while searching')
    args = parser.parse_args()
    if args.max_frames < 0 or min(args.save_every, args.redetect_every) < 1:
        parser.error('Invalid frame limits.')
    if not args.video.is_file():
        parser.error(f'Video not found: {args.video}')
    params = load_camera_params(str(args.camera_params)) if args.camera_params else None
    if params is not None and any(params.get(k) is None for k in ('camera_matrix','dist_coeffs','image_size')):
        parser.error('Calibration must include camera_matrix, dist_coeffs and image_size.')
    output = args.output / args.video.stem
    output.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        parser.error('Cannot open video.')
    fps = capture.get(cv2.CAP_PROP_FPS)
    tracker, detector = LandingPadTracker(), SlowDetector()
    rows, maps = [], None
    next_detection, attempts, losses, initializations = 0, 0, 0, 0
    index = 0
    try:
        while args.max_frames == 0 or index < args.max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            start = perf_counter()
            height, width = frame.shape[:2]
            if params is not None:
                if tuple(params['image_size']) != (width, height):
                    raise ValueError(f'Calibration size {tuple(params["image_size"])} != input {(width,height)}')
                if maps is None:
                    K = params['camera_matrix']
                    maps = cv2.initUndistortRectifyMap(K, params['dist_coeffs'], None, K,
                                                     (width,height), cv2.CV_32FC1)
                frame = cv2.remap(frame, maps[0], maps[1], cv2.INTER_LINEAR)
            reason, source, state = 'waiting_for_detection', None, 'SEARCHING'
            failure, tracking, tracking_ms, detection_ms = None, None, None, None
            inlier_ratio, points, lk_error = None, 0, None
            if tracker.initialized:
                tick = perf_counter()
                tracking = tracker.update(frame)
                tracking_ms = (perf_counter()-tick)*1000
                reason = tracking.reason
                if tracking.valid:
                    source, state = 'tracker', 'TRACKING'
                    points, inlier_ratio = tracking.num_points, tracking.inlier_ratio
                    lk_error = tracking.mean_lk_error
                else:
                    losses += 1
                    state, failure = 'LOST', tracking.reason
            if not tracker.initialized and index >= next_detection:
                tick = perf_counter()
                detection = detector.detect(frame)
                detection_ms = (perf_counter()-tick)*1000
                attempts += 1
                next_detection = index + args.redetect_every
                if detection.valid and tracker.initialize(frame, detection.corners):
                    initializations += 1
                    source, state, reason = 'detector', 'TRACKING', 'initialized'
                    points = len(tracker.prev_points)
                else:
                    reason = 'not_enough_features' if detection.valid else 'pad_not_detected'
            valid = tracker.initialized
            corners = tracker.corners.copy() if valid else None
            elapsed = (perf_counter()-start)*1000
            preview_path = None
            if index % args.save_every == 0 or reason == 'initialized' or failure is not None:
                scale = min(1., 1280/width)
                preview = cv2.resize(frame, (round(width*scale), round(height*scale)), interpolation=cv2.INTER_AREA)
                if valid:
                    cv2.polylines(preview, [np.round(corners*scale).astype(np.int32)], True, (0,220,0), 2)
                    xy = tracker.prev_points.reshape(-1,2)*tracker.scale_xy*scale
                    for x,y in xy:
                        cv2.circle(preview, (round(float(x)),round(float(y))), 2, (0,0,255), -1)
                color = (0,220,0) if valid else (0,0,255)
                cv2.putText(preview, f'{index}: {state} | {reason} | points={points}', (12,26),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
                preview_path = output / f'frame_{index:06d}.jpg'
                if not cv2.imwrite(str(preview_path), preview):
                    raise OSError(f'Cannot write {preview_path}')
            rows.append(dict(frame_id=index, timestamp_s=index/fps if fps>0 else None,
                             valid=valid, state=state, source=source, reason=reason,
                             tracking_failure=failure, corners=None if corners is None else corners.tolist(),
                             num_points=points, inlier_ratio=inlier_ratio, mean_lk_error=lk_error,
                             processing_ms=elapsed, tracking_ms=tracking_ms, detection_ms=detection_ms,
                             preview_image=str(preview_path) if preview_path else None))
            if index % args.save_every == 0 or failure is not None or reason == 'initialized':
                print(f'frame_{index:06d}: {state}, {reason}, points={points}, {elapsed:.1f} ms', flush=True)
            index += 1
    except (ValueError, OSError, cv2.error) as error:
        parser.exit(1, f'Tracking failed: {error}\n')
    finally:
        capture.release()
    if not rows:
        parser.error('No readable frames.')
    report = dict(source=str(args.video), camera_params=str(args.camera_params) if params is not None else None,
                  coordinate_space='undistorted_original_resolution' if params is not None else 'input_original_resolution',
                  frame_size=[width,height], processed_frames=len(rows), valid_frames=sum(r['valid'] for r in rows),
                  detection_attempts=attempts, initializations=initializations, tracking_losses=losses,
                  tracker_frames=sum(r['source']=='tracker' for r in rows), frames=rows)
    path = output / 'results.json'
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(f'Processed {len(rows)} consecutive frames; valid={report["valid_frames"]}; '
          f'detection attempts={attempts}; tracking losses={losses}. Saved {path}')


if __name__ == '__main__':
    main()
