import json
from pathlib import Path
import sys
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geometry.pose import estimate_pose

out = Path('outputs/pose_diagnosis')
out.mkdir(parents=True, exist_ok=True)
params = np.load('calibration/camera_params.npz')
K, dist = params['camera_matrix'], params['dist_coeffs']
report = json.loads(Path('outputs/tracking_x_verified/landing_pad2/results.json').read_text())
rows = [r for r in report['frames'] if r['valid']]
sample = [rows[i] for i in np.linspace(0, len(rows)-1, 18).astype(int)]
unit = np.float32([[0,0],[1,0],[1,1],[0,1]])
derived = []
for row in rows:
    H = cv2.getPerspectiveTransform(unit, np.float32(row['corners']))
    centred = np.float64([[1,0,-K[0,2]],[0,1,-K[1,2]],[0,0,1]]) @ H
    a, b = centred[:,0], centred[:,1]
    denom = a[2]*b[2]
    f2 = -np.dot(a[:2],b[:2])/denom if abs(denom)>1e-12 else -1
    if f2 > 0:
        f = float(np.sqrt(f2))
        scale = np.array([1/f,1/f,1])
        ratio = float(np.linalg.norm(a*scale)/np.linalg.norm(b*scale))
        derived.append(dict(frame_id=row['frame_id'], focal_px=f, short_long_ratio=min(ratio,1/ratio)))
profile = []
for f in np.linspace(1000,6500,56):
    trial = K.copy()
    trial[0,0] = trial[1,1] = f
    errors = [estimate_pose(row['corners'], trial, None, max_reprojection_error_px=100).reprojection_error_px for row in sample]
    profile.append(dict(focal_px=float(f), median_error_px=float(np.median(errors)),
                        rms_error_px=float(np.sqrt(np.mean(np.square(errors))))))
summary = dict(derived_count=len(derived),
               derived_focal_quantiles=np.percentile([d['focal_px'] for d in derived],[0,25,50,75,100]).tolist(),
               derived_ratio_quantiles=np.percentile([d['short_long_ratio'] for d in derived],[0,25,50,75,100]).tolist(),
               focal_profile_best=min(profile,key=lambda d:d['rms_error_px']))
(out/'initial_diagnosis.json').write_text(json.dumps(dict(summary=summary,profile=profile,derived=derived),indent=2))
print(json.dumps(summary, indent=2))
print('sample derived', derived[::70])
