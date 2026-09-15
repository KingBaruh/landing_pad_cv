import json
from pathlib import Path
import statistics

import cv2
import numpy as np
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.messages import FastResult
from common.drawing import draw_fast_result

output = Path('outputs/runtime_new_video')
report = json.loads((output/'results.json').read_text())
runtime = json.loads((output/'runtime_summary.json').read_text())
video = next(e['stats'] for e in runtime['events'] if e['worker']=='Video' and e['event']=='finished')
rows = report['frames']
assert set(runtime['exit_codes'].values()) == {0}
assert not runtime['forced_termination']
assert len({e['pid'] for e in runtime['events'] if e['event']=='started'}) == 3
for a,b in ((380,450),(720,755)):
    reviewed = [r for r in rows if a <= r['frame_id'] <= b]
    assert reviewed and all(not r['valid'] and not r['pose_valid'] for r in reviewed)
summary = dict(decoded_frames=video['decoded_frames'],processed_frames=len(rows),
               source_frames_without_fast_result=video['decoded_frames']-len(rows),
               pose_valid_frames=report['pose_valid_frames'],
               source_fps=video['source_fps'],elapsed_s=video['elapsed_s'],
               effective_processed_fps=len(rows)/report['elapsed_s'],
               median_fast_ms=statistics.median(r['processing_ms'] for r in rows),
               median_frame_to_result_ms=statistics.median(r['latency_ms'] for r in rows),
               delayed_detections_accepted=report['accepted_detections'],
               three_distinct_workers=True,clean_exit=True,
               reviewed_absence_has_no_pose=True)

synthetic = json.loads(Path('outputs/runtime_synthetic/results.json').read_text())
truth = json.loads(Path('outputs/pose_synthetic/ground_truth.json').read_text())
errors = []
for row in synthetic['frames']:
    expected = truth[row['frame_id']]
    if not expected['visible']:
        assert not row['pose_valid'] and row['distance_m'] is None
    if row['pose_valid']:
        errors.append(float(np.linalg.norm(np.array(row['position_xyz'])-expected['position_xyz_m'])))
assert errors and max(errors) < .02
summary['synthetic_max_position_error_m'] = max(errors)
(output/'validation.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))

# Re-render a few recorded outputs with the final readable overlay panel.
selected = {r['frame_id']:r for r in rows if r['frame_id'] in (60,540,750)}
cap = cv2.VideoCapture('videos/landing_pad_test.mp4')
params = np.load('calibration/camera_params.npz')
K = np.float64(report['camera_matrix'])
size = tuple(report['working_size'])
maps = cv2.initUndistortRectifyMap(K,params['dist_coeffs'],None,K,size,cv2.CV_32FC1)
for i in range(max(selected)+1):
    if not cap.grab(): break
    if i not in selected: continue
    ok,frame = cap.retrieve()
    assert ok
    frame = cv2.remap(cv2.resize(frame,size,interpolation=cv2.INTER_AREA),*maps,cv2.INTER_LINEAR)
    result = FastResult(**selected[i],frame=frame)
    assert cv2.imwrite(str(output/f'frame_{i:06d}.jpg'),draw_fast_result(frame.copy(),result))
cap.release()
