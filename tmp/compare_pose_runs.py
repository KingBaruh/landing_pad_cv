import json
from pathlib import Path
import cv2
import numpy as np

out=Path('outputs/pose_comparison')
out.mkdir(parents=True,exist_ok=True)
table_old=Path('outputs/table_pose_check/landing_pad_table')
table_new=Path('outputs/table_pose_final/landing_pad_table')
bed_new=Path('outputs/bed_pose_final/landing_pad2')
def metrics(folder):
    data=json.loads((folder/'results.json').read_text())
    valid=[r for r in data['frames'] if r['pose']['valid']]
    ids=[r['frame_id'] for r in valid]
    groups=np.split(np.array(ids), np.where(np.diff(ids)>1)[0]+1)
    return dict(processed_frames=data['processed_frames'],reported_tracking_valid=data['valid_frames'],
                pose_valid_frames=len(valid),pose_valid_percent=100*len(valid)/data['processed_frames'],
                tracking_losses=data['tracking_losses'],detection_attempts=data['detection_attempts'],
                pose_rms_min_median_max_px=np.percentile([r['pose']['reprojection_error_px'] for r in valid],[0,50,100]).tolist(),
                pose_valid_intervals=[[int(g[0]),int(g[-1])] for g in groups if len(g)])
summary=dict(table_before=metrics(table_old),table_after=metrics(table_new),bed_after=metrics(bed_new),
             bed_before=dict(source='outputs/pose_check/landing_pad2/pose_audit.json',pose_valid_frames=0,processed_frames=794),
             threshold_px=5,unit_tests_passed=37,checkerboard_negatives=47,checkerboard_false_positives=0,
             note='Availability and fit quality on development videos; metric accuracy has not been measured independently.')
(out/'validation.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
tiles=[]
for index in (120,220,330):
    pair=[]
    for label,folder in [('BEFORE',table_old),('AFTER',table_new)]:
        frame=cv2.imread(str(folder/f'frame_{index:06d}.jpg'))
        frame=cv2.resize(frame,(640,360))
        panel=np.full((392,640,3),245,np.uint8)
        panel[32:]=frame
        cv2.putText(panel,f'{label} - frame {index}',(12,23),cv2.FONT_HERSHEY_SIMPLEX,.65,(20,20,20),2)
        pair.append(panel)
    tiles.append(np.hstack(pair))
cv2.imwrite(str(out/'table_before_after.png'),np.vstack(tiles))
print(json.dumps(summary,indent=2))
