"""Read-only sampling for human visibility review; no detector is called."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np

out=Path('outputs/heldout_test/landing_pad_test')
out.mkdir(parents=True,exist_ok=True)
paths=[Path('videos/landing_pad_test.mp4'),Path('calibration/camera_params.npz')]
paths += [p for folder in ('detection','tracking','geometry') for p in Path(folder).glob('*.py')]
hashes={}
for path in paths:
    with path.open('rb') as stream:
        hashes[str(path)]=hashlib.file_digest(stream,'sha256').hexdigest()
(out/'evaluated_files_sha256.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
capture=cv2.VideoCapture('videos/landing_pad_test.mp4')
index=0;tiles=[]
while True:
    ok=capture.grab()
    if not ok:break
    if index%30==0:
        ok,frame=capture.retrieve()
        if not ok:raise RuntimeError(index)
        small=cv2.resize(frame,(960,540),interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out/f'presence_{index:06d}.png'),small)
        tile=np.full((294,480,3),245,np.uint8)
        tile[24:]=cv2.resize(small,(480,270))
        cv2.putText(tile,str(index),(10,18),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),1)
        tiles.append(tile)
    index+=1
capture.release()
for start in range(0,len(tiles),12):
    batch=tiles[start:start+12]
    while len(batch)<12:batch.append(np.full_like(tiles[0],245))
    cv2.imwrite(str(out/f'presence_page_{start//12}.png'),np.vstack([np.hstack(batch[j:j+4]) for j in range(0,12,4)]))
print('Reviewed input samples:',len(tiles),'decoded frames:',index)
