from pathlib import Path
import cv2
import numpy as np

out = Path('outputs/six_experiment_review')
examples = [(1,90),(2,90),(3,180),(4,120),(5,120),(6,150),(5,210),(6,210),(6,331)]
tiles=[]
for n,i in examples:
    path=Path(f'outputs/experiment_v{n}/frame_{i:06d}.jpg')
    frame=cv2.imread(str(path))
    assert frame is not None,str(path)
    tile=np.full((390,640,3),245,np.uint8)
    tile[30:]=cv2.resize(frame,(640,360),interpolation=cv2.INTER_AREA)
    cv2.putText(tile,f'v{n} | saved result frame {i}',(10,22),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),1)
    tiles.append(tile)
assert cv2.imwrite(str(out/'overlay_review.jpg'),np.vstack([np.hstack(tiles[i:i+3]) for i in (0,3,6)]))
