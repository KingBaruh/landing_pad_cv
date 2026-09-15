import cv2
import numpy as np
from pathlib import Path

out=Path('outputs/heldout_test/landing_pad_test')
ranges=[(340,390),(450,500),(665,720),(750,810)]
cap=cv2.VideoCapture('videos/landing_pad_test.mp4')
index=0; pages=[[] for _ in ranges]
while cap.grab():
    for j,(lo,hi) in enumerate(ranges):
        if lo<=index<=hi and (index-lo)%5==0:
            ok,frame=cap.retrieve()
            if not ok:raise RuntimeError(index)
            small=cv2.resize(frame,(480,270),interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out/f'transition_{index:06d}.png'),small)
            tile=np.full((294,480,3),245,np.uint8);tile[24:]=small
            cv2.putText(tile,str(index),(8,18),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),1)
            pages[j].append(tile)
    index+=1
    if index>810:break
cap.release()
for j,page in enumerate(pages):
    while len(page)%4:page.append(np.full((294,480,3),245,np.uint8))
    cv2.imwrite(str(out/f'transitions_{j}.png'),np.vstack([np.hstack(page[k:k+4]) for k in range(0,len(page),4)]))
print('Transition review saved.')
