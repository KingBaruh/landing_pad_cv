"""Sequential Day 2 tracker exercise: python -m tracking.demo --help.

Processes every frame; --save-every only controls saved preview images.
Use --pose with matching calibration for metric A4 pose estimation.
This is an offline runner, not the future three-process application.
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
from tracking.recovery import detect_for_recovery, RecoverySchedule
from geometry.homography import rectify_pad
from geometry.pose import estimate_pose
from config.config import AppConfig


def pose_record(pose):
    if pose is None:
        return None
    return dict(valid=pose.valid, reason=pose.reason,
                position_xyz_m=None if pose.position_xyz is None else pose.position_xyz.tolist(),
                distance_m=pose.distance_m,
                orientation_rpy_rad=None if pose.orientation_rpy is None else pose.orientation_rpy.tolist(),
                orientation_rpy_deg=None if pose.orientation_rpy is None else np.rad2deg(pose.orientation_rpy).tolist(),
                reprojection_error_px=pose.reprojection_error_px,
                rvec=None if pose.rvec is None else pose.rvec.ravel().tolist(),
                tvec=None if pose.tvec is None else pose.tvec.ravel().tolist(),
                corner_indices=None if pose.corner_indices is None else pose.corner_indices.tolist(),
                orientation_ambiguous=pose.orientation_ambiguous,
                yaw_ambiguous_180=pose.yaw_ambiguous_180)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--camera-params', type=Path)
    parser.add_argument('--pose', action='store_true', help='Estimate A4 pose; requires matching --camera-params')
    parser.add_argument('--output', type=Path, default=Path('outputs/tracking'))
    parser.add_argument('--debug', action='store_true', help='Save crops and proposed corners for X checks')
    parser.add_argument('--max-frames', type=int, default=0, help='0 processes the whole video')
    parser.add_argument('--save-every', type=int, default=30, help='Save a preview every N frames')
    parser.add_argument('--redetect-every', type=int, default=30,
                        help='Interval between searches allowed to fall back to the full frame')
    parser.add_argument('--local-retry-every', type=int, default=5,
                        help='Retry a recently lost pad locally every N frames')
    parser.add_argument('--pose-reset-after', type=int, default=3,
                        help='With --pose, reset tracking after N consecutive rejected tracked poses')
    args = parser.parse_args()
    if args.max_frames < 0 or min(args.save_every, args.redetect_every, args.local_retry_every,
                                  args.pose_reset_after) < 1:
        parser.error('Invalid frame limits.')
    if args.pose and args.camera_params is None:
        parser.error('--pose requires --camera-params; focal length cannot be guessed.')
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
    tracker = LandingPadTracker()
    detector = SlowDetector(camera_matrix=params['camera_matrix'] if args.pose else None,
                            max_pose_error_px=AppConfig().max_reprojection_error_px)
    rows, maps = [], None
    schedule = RecoverySchedule(args.redetect_every, args.local_retry_every)
    attempts, losses, initializations = 0, 0, 0
    last_corners, detector_calls = None, 0
    last_valid_frame = None
    previous_pose = None
    pose_rejection_streak = 0
    pose_error_limit = AppConfig().max_reprojection_error_px
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
            detection_scope = None
            recovery_mode = None
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
            if not tracker.initialized:
                recovery_mode = schedule.due(index, last_valid_frame)
            if recovery_mode is not None:
                tick = perf_counter()
                detection, detection_scope, calls = detect_for_recovery(
                    frame, detector, last_corners, allow_global=recovery_mode == 'full')
                detector_calls += calls
                detection_ms = (perf_counter()-tick)*1000
                attempts += int(calls > 0)
                schedule.attempted(index, recovery_mode)
                if detection.valid and tracker.initialize(frame, detection.corners):
                    initializations += 1
                    source, state, reason = 'detector', 'TRACKING', 'initialized'
                    points = len(tracker.prev_points)
                else:
                    reason = 'not_enough_features' if detection.valid else 'pad_not_detected'
                    if calls == 0:
                        reason = 'waiting_for_full_search'
                        recovery_mode = None
                    elif not detection.valid and detector.last_pose_rejections:
                        reason = 'no_pose_consistent_detection'
            valid = tracker.initialized
            corners = tracker.corners.copy() if valid else None
            rejected_tracking_corners = None
            pose, pose_ms = None, None
            if args.pose:
                if failure is not None or source == 'detector' or not valid:
                    previous_pose = None
                tick = perf_counter()
                # Frames were remapped with the same K: do NOT apply distortion again.
                pose = estimate_pose(corners, K, None, previous=previous_pose,
                                     max_reprojection_error_px=pose_error_limit)
                pose_ms = (perf_counter()-tick)*1000
                previous_pose = pose if pose.valid else None
                # LK/H agreement and a visible X do not establish that corners
                # still describe a calibrated A4 rectangle. Reacquire from image
                # evidence after sustained failure; never relax the pose limit.
                pose_rejection_streak = (pose_rejection_streak+1
                                         if valid and source == 'tracker' and not pose.valid else 0)
                if pose_rejection_streak >= args.pose_reset_after:
                    rejected_tracking_corners = corners.tolist()
                    tracker.reset()
                    losses += 1
                    valid, corners, points = False, None, 0
                    state, reason, failure = 'LOST', 'pose_inconsistent', 'pose_inconsistent'
                    pose_rejection_streak = 0
            if valid:
                last_corners = corners.copy()
                last_valid_frame = index
            elapsed = (perf_counter()-start)*1000
            x_check = tracking.x_check if tracking is not None else None
            if args.debug and x_check is not None:
                proposed = np.float32(x_check['candidate_corners'])
                crop_path = output / f'frame_{index:06d}_x_crop.png'
                candidate_path = output / f'frame_{index:06d}_x_candidate.jpg'
                crop = rectify_pad(frame, proposed, (280,396))
                preview_scale = min(1., 1280/width)
                candidate = cv2.resize(frame, (round(width*preview_scale), round(height*preview_scale)))
                cv2.polylines(candidate, [np.round(proposed*preview_scale).astype(np.int32)],
                              True, (0,190,255), 2)
                if not cv2.imwrite(str(crop_path), crop) or not cv2.imwrite(str(candidate_path), candidate):
                    raise OSError('Cannot save X diagnostics.')
                x_check = dict(x_check, crop_image=str(crop_path), candidate_image=str(candidate_path))
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
                if pose is not None:
                    if pose.valid:
                        preview_K = K.copy()
                        preview_K[:2] *= scale
                        cv2.drawFrameAxes(preview, preview_K, None, pose.rvec, pose.tvec, .07, 2)
                        x, y, z = pose.position_xyz
                        roll, pitch, yaw = np.rad2deg(pose.orientation_rpy)
                        lines = [f'Distance={pose.distance_m:.3f} m | XYZ=({x:.3f}, {y:.3f}, {z:.3f}) m',
                                 f'RPY=({roll:.1f}, {pitch:.1f}, {yaw:.1f}) deg | yaw mod 180',
                                 f'Pose RMS={pose.reprojection_error_px:.2f} px | alternate pose={pose.orientation_ambiguous}']
                    else:
                        lines = [f'Pose unavailable: {pose.reason}']
                        if pose.reprojection_error_px is not None:
                            lines.append(f'Pose RMS={pose.reprojection_error_px:.2f} px (limit {pose_error_limit:g} px)')
                    panel_width = min(preview.shape[1]-1, max(cv2.getTextSize(
                        line, cv2.FONT_HERSHEY_SIMPLEX, .55, 1)[0][0] for line in lines)+24)
                    cv2.rectangle(preview, (4,34), (panel_width, 60+24*(len(lines)-1)), (0,0,0), -1)
                    for line_index, line in enumerate(lines):
                        origin = (12, 52+24*line_index)
                        cv2.putText(preview, line, origin, cv2.FONT_HERSHEY_SIMPLEX, .55, (255,255,255), 1)
                preview_path = output / f'frame_{index:06d}.jpg'
                if not cv2.imwrite(str(preview_path), preview):
                    raise OSError(f'Cannot write {preview_path}')
            rows.append(dict(frame_id=index, timestamp_s=index/fps if fps>0 else None,
                             valid=valid, state=state, source=source, reason=reason,
                             tracking_failure=failure, corners=None if corners is None else corners.tolist(),
                             num_points=points, inlier_ratio=inlier_ratio, mean_lk_error=lk_error,
                             processing_ms=elapsed, tracking_ms=tracking_ms, detection_ms=detection_ms,
                             pose=pose_record(pose), pose_ms=pose_ms,
                             pose_rejection_streak=pose_rejection_streak,
                             rejected_tracking_corners=rejected_tracking_corners,
                             detection_scope=detection_scope,
                             recovery_mode=recovery_mode,
                             x_check=x_check,
                             preview_image=str(preview_path) if preview_path else None))
            if index % args.save_every == 0 or failure is not None or reason == 'initialized':
                pose_text = '' if pose is None else (f', distance={pose.distance_m:.3f} m, pose RMS={pose.reprojection_error_px:.2f} px'
                                                     if pose.valid else f', pose={pose.reason}')
                print(f'frame_{index:06d}: {state}, {reason}, points={points}, {elapsed:.1f} ms{pose_text}', flush=True)
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
                  detector_calls=detector_calls,
                  full_search_attempts=sum(r['recovery_mode']=='full' for r in rows),
                  local_retry_attempts=sum(r['recovery_mode']=='local' for r in rows),
                  pose_enabled=args.pose,
                  pose_reset_after=args.pose_reset_after if args.pose else None,
                  pose_triggered_resets=sum(r['tracking_failure']=='pose_inconsistent' for r in rows),
                  pose_valid_frames=sum(r['pose'] is not None and r['pose']['valid'] for r in rows),
                  pose_convention='A4 object to camera; metres; camera x right, y down, z forward; R=Rz(yaw)Ry(pitch)Rx(roll); yaw ambiguous by 180 degrees' if args.pose else None,
                  tracker_frames=sum(r['source']=='tracker' for r in rows), frames=rows)
    path = output / 'results.json'
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(f'Processed {len(rows)} consecutive frames; valid={report["valid_frames"]}; '
          f'detection attempts={attempts}; tracking losses={losses}. Saved {path}')


if __name__ == '__main__':
    main()
