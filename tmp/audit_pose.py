import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibration.calibrate import load_camera_params
from geometry.pose import estimate_pose
from tracking.demo import pose_record

source = Path('outputs/tracking_x_verified/landing_pad2/results.json')
report = json.loads(source.read_text(encoding='utf-8'))
assert report['coordinate_space'] == 'undistorted_original_resolution'
params = load_camera_params('calibration/camera_params.npz')
assert list(params['image_size']) == report['frame_size']
rows, previous = [], None
for row in report['frames']:
    if row['tracking_failure'] is not None or row['source'] == 'detector' or not row['valid']:
        previous = None
    pose = estimate_pose(row['corners'], params['camera_matrix'], None, previous=previous)
    previous = pose if pose.valid else None
    rows.append(dict(frame_id=row['frame_id'], tracking_valid=row['valid'], pose=pose_record(pose)))
errors = [r['pose']['reprojection_error_px'] for r in rows if r['tracking_valid']]
summary = dict(source=str(source), audit_kind='pose_only_on_saved_undistorted_corners',
               processed_frames=len(rows), tracked_frames=sum(r['tracking_valid'] for r in rows),
               pose_valid_frames=sum(r['pose']['valid'] for r in rows),
               reprojection_rms_min_median_max_px=np.percentile(errors,[0,50,100]).tolist())
path = Path('outputs/pose_check/landing_pad2/pose_audit.json')
path.write_text(json.dumps(dict(summary=summary, frames=rows), indent=2, allow_nan=False), encoding='utf-8')
print(json.dumps(summary, indent=2))
