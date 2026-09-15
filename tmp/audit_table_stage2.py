"""Inspect failed development frames without changing detection thresholds."""
import json
from pathlib import Path
import sys
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from detection.slow_detector import SlowDetector
from geometry.homography import rectify_pad

out=Path('outputs/stage2_audit')
out.mkdir(parents=True,exist_ok=True)
base=json.loads(Path('outputs/table_local_recovery/landing_pad_table/results.json').read_text())
params=np.load('calibration/camera_params.npz'); K=params['camera_matrix']
maps=cv2.initUndistortRectifyMap(K,params['dist_coeffs'],None,K,(3840,2160),cv2.CV_32FC1)
capture=cv2.VideoCapture('videos/landing_pad_table.mp4')
targets=[125,130,145,165,170,180,190,195,210,225,288,292,297,302,312,317]
results=[]; previews=[]
for i in targets:
    capture.set(cv2.CAP_PROP_POS_FRAMES,i)
    ok,frame=capture.read()
    if not ok: raise RuntimeError(i)
    frame=cv2.remap(frame,*maps,cv2.INTER_LINEAR)
    detector=SlowDetector(camera_matrix=K)
    result=detector.detect(frame,debug=True)
    details=detector.candidate_diagnostics
    row=dict(frame_id=i,baseline_reason=base['frames'][i]['reason'],
             baseline_loss=base['frames'][i]['tracking_failure'],
             global_valid=result.valid,candidates=details)
    if base['frames'][i]['x_check']:
        row['tracking_x_check']=base['frames'][i]['x_check']
    results.append(row)
    preview=cv2.resize(frame,(640,360))
    candidates=sorted(details,key=lambda d:d['confidence'],reverse=True)[:3]
    for j,d in enumerate(candidates):
        points=np.float32(d['corners'])
        crop=rectify_pad(frame,points,(280,396))
        cv2.imwrite(str(out/f'frame_{i:06d}_candidate_{j}.png'),crop)
        cv2.polylines(preview,[np.round(points/6).astype(np.int32)],True,[(0,210,0),(0,170,255),(230,40,40)][j],1)
    cv2.putText(preview,f'{i}: global={result.valid}',(8,24),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,255),2)
    cv2.imwrite(str(out/f'frame_{i:06d}.png'),preview)
    previews.append(preview)
    print(i,'global',result.valid,'top',[(d['candidate'],round(d['confidence'],2),round(d['border_support'],2),d['pose_error_px'],d['x_check'].get('rejection_reason')) for d in candidates],flush=True)
capture.release()
(out/'audit.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
cv2.imwrite(str(out/'contact.png'),np.vstack([np.hstack(previews[j:j+4]) for j in range(0,len(previews),4)]))
