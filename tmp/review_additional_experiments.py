"""Audit new recordings and visualize saved results on their matching frames."""
import json
from pathlib import Path
import cv2
import numpy as np

out=Path('outputs/additional_experiment_review')
out.mkdir(parents=True,exist_ok=True)
summaries=[]
for name in ('v7','v8_landing'):
    root=Path('outputs')/f'experiment_{name}'
    runtime=json.loads((root/'runtime_summary.json').read_text())
    report=json.loads((root/'results.json').read_text())
    rows=report['frames']
    by_id={r['frame_id']:r for r in rows}
    cap=cv2.VideoCapture(runtime['config']['source'])
    assert cap.isOpened()
    meta=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps=cap.get(cv2.CAP_PROP_FPS)
    wanted={min(by_id,key=lambda i:abs(i-t)) for t in np.linspace(0,meta-5,12)}
    K=np.float64(report['camera_matrix'])
    dist=np.load('calibration/camera_params.npz')['dist_coeffs']
    size=tuple(report['working_size'])
    maps=cv2.initUndistortRectifyMap(K,dist,None,K,size,cv2.CV_32FC1)
    tiles=[]
    i=0
    while cap.grab():
        if i in wanted:
            ok,frame=cap.retrieve()
            assert ok
            frame=cv2.remap(cv2.resize(frame,size,interpolation=cv2.INTER_AREA),*maps,cv2.INTER_LINEAR)
            row=by_id[i]
            if row['corners'] is not None:
                cv2.polylines(frame,[np.round(row['corners']).astype(np.int32)],True,
                              (0,220,0) if row['pose_valid'] else (0,150,255),3)
            tile=np.full((305,480,3),245,np.uint8)
            tile[35:]=cv2.resize(frame,(480,270),interpolation=cv2.INTER_AREA)
            status='Pose OK' if row['pose_valid'] else 'no Pose'
            cv2.putText(tile,f'{name} | {i} | {i/fps:.2f}s | {status}',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1)
            tiles.append(tile)
        i+=1
    cap.release()
    while len(tiles)<12:tiles.append(np.full((305,480,3),245,np.uint8))
    cv2.imwrite(str(out/f'{name}_contact.jpg'),np.vstack([np.hstack(tiles[j:j+3]) for j in (0,3,6,9)]))
    groups=[]
    group=[]
    for row in rows:
        if not row['pose_valid']:group.append(row)
        elif group:groups.append(group);group=[]
    if group:groups.append(group)
    gaps=[dict(first=g[0]['frame_id'],last=g[-1]['frame_id'],start_s=g[0]['timestamp'],end_s=g[-1]['timestamp'],
               reasons=sorted({r['reason'] for r in g})) for g in groups]
    accepted=[r for r in rows if r['pose_valid']]
    stats=next(e['stats'] for e in runtime['events'] if e['event']=='finished' and e['worker']=='Video')
    summary=dict(video=name,source=runtime['config']['source'],full_sequential_decoded=i,metadata_frames=meta,
                 runtime_decoded=stats['decoded_frames'],processed=len(rows),pose_valid=len(accepted),
                 tracking_valid=report['valid_frames'],losses=report['losses'],pose_resets=report['pose_resets'],
                 first_pose_frame=accepted[0]['frame_id'] if accepted else None,
                 first_pose_s=accepted[0]['timestamp'] if accepted else None,
                 reported_invalid=gaps,exit_codes=runtime['exit_codes'],stopped=runtime['stop_requested'])
    summaries.append(summary)
    print(json.dumps(summary),flush=True)
(out/'audit.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
