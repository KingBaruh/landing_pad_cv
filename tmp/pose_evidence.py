import json
from pathlib import Path
import sys
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geometry.pose import estimate_pose

out=Path('outputs/pose_diagnosis')
samples=json.loads((out/'sample_diagnosis.json').read_text())
rows=json.loads(Path('outputs/tracking_x_verified/landing_pad2/results.json').read_text())['frames']
first=samples[0];origin=np.array(first['crop_origin'])
image=cv2.imread(str(out/'frame_000000_clean.png'))
corners=np.float64(rows[0]['corners'])-origin
hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
mask=np.uint8((hsv[:,:,1]<10)&(hsv[:,:,2]>100))*255
contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
contour=max(contours,key=cv2.contourArea).reshape(-1,2)
a,b=corners[0],corners[3]
edge=b-a;length=np.linalg.norm(edge);normal=np.array([-edge[1],edge[0]])/length
fraction=(contour-a) @ edge / length**2
offset=(contour-a) @ normal
selected=(fraction>.15)&(fraction<.85)&(np.abs(offset)<50)
curve=contour[selected]; offsets=offset[selected]
for point in curve:
    cv2.circle(image,tuple(point),1,(255,190,0),-1)
cv2.line(image,tuple(a.astype(int)),tuple(b.astype(int)),(0,255,255),2)
worst=curve[np.argmax(np.abs(offsets))];foot=worst-normal*((worst-a) @ normal)
cv2.line(image,tuple(worst.astype(int)),tuple(foot.astype(int)),(0,0,255),3)
deviation=float(np.percentile(np.abs(offsets),95))
cv2.imwrite(str(out/'curved_edge.png'),image)

refined=json.loads((out/'refinement_diagnosis.json').read_text())
case=next(m for m in refined if m['frame_id']==120)
origin120=np.array(next(m for m in samples if m['frame_id']==120)['crop_origin'])
manual=np.array(case['trials'][-1]['corners'])-origin120
tracked=np.float64(rows[120]['corners'])-origin120
image120=cv2.imread(str(out/'frame_000120_clean.png'))
for old,new in zip(tracked,manual):
    before=tuple(np.round(old).astype(int));after=tuple(np.round(new).astype(int))
    cv2.circle(image120,before,7,(0,0,255),2)
    cv2.circle(image120,after,7,(0,200,0),2)
    cv2.arrowedLine(image120,before,after,(0,160,255),2,tipLength=.2)
cv2.imwrite(str(out/'tracked_vs_manual.png'),image120)

def panel(image,title,subtitle):
    scale=880/image.shape[1]
    small=cv2.resize(image,(880,round(image.shape[0]*scale)),interpolation=cv2.INTER_AREA)
    canvas=np.full((small.shape[0]+72,880,3),245,np.uint8)
    canvas[72:]=small
    cv2.putText(canvas,title,(12,27),cv2.FONT_HERSHEY_SIMPLEX,.7,(20,20,20),2)
    cv2.putText(canvas,subtitle,(12,54),cv2.FONT_HERSHEY_SIMPLEX,.52,(20,20,20),1)
    return canvas
top=panel(image,'Frame 0: curved paper boundary',f'Yellow: straight chord. Cyan: measured edge. 95th-percentile bow: {deviation:.1f} px.')
bottom=panel(image120,'Frame 120: tracking corner drift','Red: tracked. Green: visually selected physical corners. RMS: 16.94 -> 11.52 px.')
cv2.imwrite(str(out/'diagnosis_evidence.jpg'),np.vstack((top,bottom)))

# Map undistorted corners back through the saved distortion model and solve
# directly in raw coordinates. This tests consistency of the coordinate route.
p=np.load('calibration/camera_params.npz');K,dist=p['camera_matrix'],p['dist_coeffs']
distortion_check=[]
for s in samples:
    pts=np.float64(rows[s['frame_id']]['corners'])
    normalized=cv2.undistortPoints(pts[:,None],K,None).reshape(4,2)
    rays=np.column_stack((normalized,np.ones(4)))
    raw=cv2.projectPoints(rays,np.zeros(3),np.zeros(3),K,dist)[0].reshape(4,2)
    fit=estimate_pose(raw,K,dist,max_reprojection_error_px=100)
    distortion_check.append(dict(frame_id=s['frame_id'],undistorted_rms=s['current_error'],
                                  raw_with_distortion_rms=fit.reprojection_error_px))
summary=dict(edge_bow_95th_percentile_px=deviation,
             manual_frame120_corner_displacements_px=case['trials'][-1]['displacement_px'],
             distortion_coordinate_check=distortion_check)
(out/'evidence_measurements.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
