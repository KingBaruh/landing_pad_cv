import json
from pathlib import Path
import sys
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from geometry.pose import estimate_pose, A4_OBJECT_POINTS

out=Path('outputs/pose_diagnosis')
samples=json.loads((out/'sample_diagnosis.json').read_text())
rows=json.loads(Path('outputs/tracking_x_verified/landing_pad2/results.json').read_text())['frames']
p=np.load('calibration/camera_params.npz');K=p['camera_matrix']
measurements=[]
for item in samples:
    index=item['frame_id'];origin=np.array(item['crop_origin'])
    frame=cv2.imread(str(out/f'frame_{index:06d}_clean.png'))
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    old=np.float32(rows[index]['corners'])
    trials=[]
    for window in (11,21,41):
        local=(old-origin).astype(np.float32).reshape(-1,1,2)
        refined=cv2.cornerSubPix(gray,local,(window,window),(-1,-1),
                                (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_MAX_ITER,100,.001)).reshape(4,2)+origin
        fit=estimate_pose(refined,K,None,max_reprojection_error_px=100)
        trials.append(dict(window=window,corners=refined.tolist(),
                           displacement_px=np.linalg.norm(refined-old,axis=1).tolist(),rms_px=fit.reprojection_error_px))
    if index==120:
        manual=np.float64([[551,59],[1162,308],[753,1072],[41,795]])+origin
        fit=estimate_pose(manual,K,None,max_reprojection_error_px=100)
        trials.append(dict(window='manual_visual',corners=manual.tolist(),
                           displacement_px=np.linalg.norm(manual-old,axis=1).tolist(),rms_px=fit.reprojection_error_px))
    best=trials[1]
    refined=np.array(best['corners'])
    visual=frame.copy()
    for j,(before,after) in enumerate(zip(old-origin,refined-origin)):
        cv2.circle(visual,tuple(np.round(before).astype(int)),5,(0,0,255),2)
        cv2.circle(visual,tuple(np.round(after).astype(int)),5,(0,200,0),2)
    cv2.imwrite(str(out/f'frame_{index:06d}_refinement.png'),visual)
    # Compare the independent globally optimal SQPnP solver with ITERATIVE.
    alternatives=[]
    for shift in range(4):
        inp=np.ascontiguousarray(np.roll(old,-shift,axis=0),dtype=np.float64)
        ok,rv,tv=cv2.solvePnP(A4_OBJECT_POINTS,inp,K,None,flags=cv2.SOLVEPNP_SQPNP)
        if ok:
            proj=cv2.projectPoints(A4_OBJECT_POINTS,rv,tv,K,None)[0].reshape(4,2)
            alternatives.append(float(np.sqrt(np.mean(np.sum((proj-inp)**2,axis=1)))))
    measurements.append(dict(frame_id=index,original_rms=item['current_error'],trials=trials,
                             sqpnp_rms=min(alternatives)))
(out/'refinement_diagnosis.json').write_text(json.dumps(measurements,indent=2))
for m in measurements:
    print(m['frame_id'],'original',round(m['original_rms'],2),'SQPNP',round(m['sqpnp_rms'],2),
          'refined',[(x['window'],round(x['rms_px'],2),round(max(x['displacement_px']),2)) for x in m['trials']])
