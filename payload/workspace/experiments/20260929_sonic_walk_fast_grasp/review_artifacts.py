"""Save sampled visual evidence; full-decode validation is a separate prerequisite."""
from pathlib import Path
import argparse,json
import cv2,numpy as np

p=argparse.ArgumentParser();p.add_argument('run',type=Path);r=p.parse_args().run
v=json.loads((r/'validation.json').read_text())
last=(int(v['videos']['head.mp4']['nb_frames'])-1)/25
times=sorted(set([0.,2.,4.,5.5,8.5,9.7,12.1,13.6,14.8,16.3,18.1,last]))
times=[t for t in times if t<=last];rows=[]
for t in times:
 columns=[]
 for name in ['head.mp4','000000.mp4']:
  cap=cv2.VideoCapture(str(r/name));index=round(t*25);cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=cap.read();cap.release();assert ok
  if t==last:cv2.imwrite(str(r/('final_'+name.replace('.mp4','.png'))),frame)
  ratio=min(480/frame.shape[1],360/frame.shape[0]);frame=cv2.resize(frame,(round(frame.shape[1]*ratio),round(frame.shape[0]*ratio)))
  canvas=np.zeros((390,480,3),np.uint8);canvas[30:30+frame.shape[0],:frame.shape[1]]=frame
  cv2.putText(canvas,f'{name}  frame{index}  t={t:.2f}s',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);columns.append(canvas)
 rows.append(np.concatenate(columns,axis=1))
cv2.imwrite(str(r/'review_montage.jpg'),np.concatenate(rows,axis=0))
(r/'review_frame_indices.json').write_text(json.dumps({'times_s':times,'video_frames':[round(t*25) for t in times]},indent=2)+'\n')
print(r/'review_montage.jpg')
