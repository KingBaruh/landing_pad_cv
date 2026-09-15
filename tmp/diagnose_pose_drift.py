"""Compare frozen tracked corners against fresh image detections on identical frames."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from calibration.calibrate import load_camera_params
from detection.slow_detector import SlowDetector
from tracking.recovery import detect_for_recovery
from geometry.pose import estimate_pose

out = Path('outputs/pose_drift_diagnosis')
out.mkdir(parents=True, exist_ok=True)
baseline = json.loads(Path('outputs/heldout_test/landing_pad_test/results.json').read_text())
params = load_camera_params('calibration/camera_params.npz')
K = params['camera_matrix']
maps = cv2.initUndistortRectifyMap(K, params['dist_coeffs'], None, K, (3840,2160), cv2.CV_32FC1)
detector = SlowDetector(camera_matrix=K)
selected = {60,65,75,90,95,100,525,530,540,560,575}
cap = cv2.VideoCapture('videos/landing_pad_test.mp4')
rows = []
for i in range(max(selected)+1):
    if not cap.grab(): break
    if i not in selected: continue
    ok, frame = cap.retrieve()
    assert ok
    frame = cv2.remap(frame, *maps, cv2.INTER_LINEAR)
    old = baseline['frames'][i]
    corners = np.float32(old['corners'])
    fresh, scope, calls = detect_for_recovery(frame, detector, corners, allow_global=False)
    pose = estimate_pose(fresh.corners, K, None)
    row = dict(frame=i, old_rms=old['pose']['reprojection_error_px'],
               fresh_valid=fresh.valid, fresh_pose_valid=pose.valid,
               fresh_rms=pose.reprojection_error_px, scope=scope,
               tracked_corners=corners.tolist(),
               fresh_corners=None if fresh.corners is None else fresh.corners.tolist())
    lo = np.maximum(np.floor(corners.min(axis=0)-80),0).astype(int)
    hi = np.minimum(np.ceil(corners.max(axis=0)+80),[3840,2160]).astype(int)
    crop = frame[lo[1]:hi[1], lo[0]:hi[0]].copy()
    cv2.polylines(crop, [np.round(corners-lo).astype(np.int32)], True, (0,0,255), 3)
    if fresh.valid:
        cv2.polylines(crop, [np.round(fresh.corners-lo).astype(np.int32)], True, (0,220,0), 2)
    cv2.imwrite(str(out/f'frame_{i:06d}.jpg'), crop)
    rows.append(row)
    print(json.dumps(row), flush=True)
cap.release()
(out/'comparison.json').write_text(json.dumps(rows, indent=2))
