"""Read-only source sampling and runtime audit for the six submission videos."""
import json
from pathlib import Path
import cv2
import numpy as np

out = Path('outputs/six_experiment_review')
out.mkdir(parents=True,exist_ok=True)


def spans(rows, predicate):
    groups = []
    active = []
    for row in rows:
        if predicate(row):
            active.append(row)
        elif active:
            groups.append(active)
            active=[]
    if active: groups.append(active)
    return [dict(first=g[0]['frame_id'],last=g[-1]['frame_id'],
                 start_s=g[0]['timestamp'],end_s=g[-1]['timestamp'],
                 processed_count=len(g),reasons=sorted(set(r['reason'] for r in g))) for g in groups]


summaries = []
for n in range(1,7):
    root = Path(f'outputs/experiment_v{n}')
    runtime = json.loads((root/'runtime_summary.json').read_text())
    report = json.loads((root/'results.json').read_text())
    rows = report['frames']
    by_id = {r['frame_id']:r for r in rows}
    source = runtime['config']['source']
    cap = cv2.VideoCapture(source)
    assert cap.isOpened(),source
    meta_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    wanted = set(np.linspace(0,max(0,meta_count-5),12).astype(int))
    tiles = []
    i=0
    while cap.grab():
        if i in wanted:
            ok,frame = cap.retrieve()
            assert ok
            preview = cv2.resize(frame,(480,270),interpolation=cv2.INTER_AREA)
            row = by_id.get(i)
            status = 'not processed' if row is None else ('Pose OK' if row['pose_valid'] else 'no Pose')
            tile = np.full((305,480,3),245,np.uint8)
            tile[35:]=preview
            cv2.putText(tile,f'v{n} | {i} | {i/fps:.2f}s | {status}',(9,23),
                        cv2.FONT_HERSHEY_SIMPLEX,.55,(20,20,20),1,cv2.LINE_AA)
            tiles.append(tile)
        i+=1
    cap.release()
    while len(tiles)<12: tiles.append(np.full((305,480,3),245,np.uint8))
    sheet = np.vstack([np.hstack(tiles[k:k+3]) for k in range(0,12,3)])
    assert cv2.imwrite(str(out/f'v{n}_contact.jpg'),sheet)
    accepted = [r for r in rows if r['pose_valid']]
    video_stats = next(e['stats'] for e in runtime['events'] if e['worker']=='Video' and e['event']=='finished')
    summary = dict(video=f'v{n}',source=source,full_sequential_decoded=i,metadata_frames=meta_count,
                   fps=fps,runtime_decoded=video_stats['decoded_frames'],
                   processed=len(rows),pose_valid=len(accepted),tracking_valid=report['valid_frames'],
                   losses=report['losses'],pose_resets=report['pose_resets'],
                   source_frames_without_results=video_stats['decoded_frames']-len(rows),
                   first_pose_frame=accepted[0]['frame_id'] if accepted else None,
                   first_pose_s=accepted[0]['timestamp'] if accepted else None,
                   median_rms_working_px=float(np.median([r['reprojection_error_px'] for r in accepted])) if accepted else None,
                   estimated_distance_range_m=[min(r['distance_m'] for r in accepted),max(r['distance_m'] for r in accepted)] if accepted else None,
                   no_tracking=spans(rows,lambda r:not r['valid']),
                   tracking_but_no_pose=spans(rows,lambda r:r['valid'] and not r['pose_valid']),
                   reported_invalid=spans(rows,lambda r:not r['pose_valid']),
                   exit_codes=runtime['exit_codes'],stopped=runtime['stop_requested'])
    summaries.append(summary)
    print(json.dumps(summary),flush=True)
(out/'audit.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
