import json
from pathlib import Path
import sys
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geometry.pose import estimate_pose, A4_OBJECT_POINTS
out=Path('outputs/pose_diagnosis')
p=np.load('calibration/camera_params.npz')
K,dist=p['camera_matrix'],p['dist_coeffs']
rows=json.loads(Path('outputs/tracking_x_verified/landing_pad2/results.json').read_text())['frames']
cap=cv2.VideoCapture('videos/landing_pad2.mp4')
maps=cv2.initUndistortRectifyMap(K,dist,None,K,(3840,2160),cv2.CV_32FC1)
targets=[0,59,120,136,240,330,480,600,720]
table=[]
for i in targets:
    row=rows[i]
    if not row['valid']: continue
    cap.set(cv2.CAP_PROP_POS_FRAMES,i)
    ok,frame=cap.read()
    if not ok:continue
    frame=cv2.remap(frame,*maps,cv2.INTER_LINEAR)
    points=np.array(row['corners'])
    x0,y0=np.maximum(points.min(axis=0)-60,0).astype(int)
    x1,y1=np.minimum(points.max(axis=0)+60,[3840,2160]).astype(int)
    crop=frame[y0:y1,x0:x1].copy()
    cv2.imwrite(str(out/f'frame_{i:06d}_clean.png'),crop)
    pose=estimate_pose(points,K,None,max_reprojection_error_px=100)
    projected=cv2.projectPoints(A4_OBJECT_POINTS,pose.rvec,pose.tvec,K,None)[0].reshape(4,2)
    projected=projected[np.argsort(pose.corner_indices)]
    for j,pt in enumerate(points):
        xy=tuple(np.round(pt-[x0,y0]).astype(int))
        xy2=tuple(np.round(projected[j]-[x0,y0]).astype(int))
        cv2.circle(crop,xy,4,(0,255,0),1)
        cv2.putText(crop,str(j),tuple(np.array(xy)+[7,-7]),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,180,0),2)
        cv2.line(crop,xy,xy2,(0,0,255),2)
    cv2.imwrite(str(out/f'frame_{i:06d}_corners.png'),crop)
    trials=[]
    for f in np.arange(700,6100,100):
        trial=K.copy();trial[0,0]=trial[1,1]=f
        result=estimate_pose(points,trial,None,max_reprojection_error_px=100)
        trials.append((float(result.reprojection_error_px),int(f)))
    table.append(dict(frame_id=i,crop_origin=[int(x0),int(y0)],
                      current_error=pose.reprojection_error_px,best_focal_trial=min(trials)))
cap.release()
(out/'sample_diagnosis.json').write_text(json.dumps(table,indent=2))
print(json.dumps(table,indent=2))
