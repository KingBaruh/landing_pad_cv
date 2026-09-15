"""Preserve comparable availability metrics and regressions for the pose reset fix."""
import hashlib
import json
from pathlib import Path
import statistics

out = Path('outputs/pose_drift_diagnosis')
pairs = [
    ('landing_pad_test', 'outputs/heldout_test', 'outputs/pose_recovery_check'),
    ('landing_pad_table', 'outputs/table_local_recovery', 'outputs/pose_recovery_regression'),
    ('landing_pad2', 'outputs/bed_local_recovery', 'outputs/pose_recovery_regression'),
]
def spans(ids):
    result = []
    for i in ids:
        if result and result[-1][1] == i-1: result[-1][1] = i
        else: result.append([i,i])
    return result

comparison = {}
for name, before, after in pairs:
    runs = [json.loads((Path(base)/name/'results.json').read_text()) for base in (before,after)]
    assert runs[0]['processed_frames'] == runs[1]['processed_frames']
    stats = []
    accepted = []
    for run in runs:
        rows = run['frames']
        good = {r['frame_id'] for r in rows if (r.get('pose') or {}).get('valid')}
        accepted.append(good)
        errors = [r['pose']['reprojection_error_px'] for r in rows if r['frame_id'] in good]
        stats.append(dict(frames=run['processed_frames'], tracking_valid=run['valid_frames'],
                          pose_valid=len(good), detector_calls=run['detector_calls'],
                          median_accepted_rms_px=statistics.median(errors),
                          tracking_valid_pose_rejected=sum(r['valid'] and r['frame_id'] not in good for r in rows),
                          pose_resets=run.get('pose_triggered_resets',0),
                          tracking_invalid_spans=spans(r['frame_id'] for r in rows if not r['valid'])))
    comparison[name] = dict(before=stats[0], after=stats[1],
                            newly_accepted_spans=spans(sorted(accepted[1]-accepted[0])),
                            formerly_accepted_now_rejected_spans=spans(sorted(accepted[0]-accepted[1])))
    if name == 'landing_pad_test':
        for a,b in ((380,450),(720,755)):
            assert all(not r['valid'] and not r['pose']['valid'] for r in runs[1]['frames'] if a <= r['frame_id'] <= b)
        comparison[name]['reviewed_absence_still_has_no_output'] = True

(out/'recovery_comparison.json').write_text(json.dumps(comparison,indent=2))
paths = [Path('tracking/demo.py'),Path('tests/test_pose_recovery.py'),Path('calibration/camera_params.npz')]
(out/'changed_run_files_sha256.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2))
print(json.dumps(comparison,indent=2))
