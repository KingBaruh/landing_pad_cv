import json
from pathlib import Path
import cv2
import numpy as np

out = Path('outputs/recovery_comparison')
out.mkdir(parents=True, exist_ok=True)
folders = {
    'table_before': Path('outputs/table_pose_final/landing_pad_table'),
    'table_after': Path('outputs/table_local_recovery/landing_pad_table'),
    'bed_before': Path('outputs/bed_pose_final/landing_pad2'),
    'bed_after': Path('outputs/bed_local_recovery/landing_pad2'),
}
summary = {}
for label, folder in folders.items():
    report = json.loads((folder/'results.json').read_text())
    rows = report['frames']
    accepted = [r for r in rows if r['pose']['valid']]
    summary[label] = dict(processed_frames=len(rows), pose_valid_frames=len(accepted),
                         pose_valid_percent=100*len(accepted)/len(rows),
                         detector_calls=report['detector_calls'],
                         actual_full_frame_calls=sum(r['detection_scope']=='global' for r in rows),
                         tracking_losses=report['tracking_losses'],
                         median_pose_rms_px=float(np.median([r['pose']['reprojection_error_px'] for r in accepted])),
                         report=str(folder/'results.json'))
summary['validation'] = dict(tracking_tests=14, pose_tests=12,
    synthetic_frames=45, synthetic_accepted_poses=35,
    note='Unchanged pose threshold of 5 px. These recordings are development data; metric accuracy was not measured independently.')
(out/'validation.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
panels = []
for frame_id in (150,270):
    pair = []
    for label in ('table_before','table_after'):
        frame = cv2.imread(str(folders[label]/f'frame_{frame_id:06d}.jpg'))
        tile = np.full((392,640,3),245,np.uint8)
        tile[32:] = cv2.resize(frame,(640,360))
        cv2.putText(tile,f'{label} | frame {frame_id}',(12,23),cv2.FONT_HERSHEY_SIMPLEX,.6,(20,20,20),2)
        pair.append(tile)
    panels.append(np.hstack(pair))
cv2.imwrite(str(out/'before_after.png'),np.vstack(panels))
print(json.dumps(summary, indent=2))
