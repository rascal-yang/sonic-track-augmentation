from pathlib import Path
import json,sys
import cv2
import numpy as np
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parents[1];r=B/'runs'/sys.argv[1];out=r/'visual_review';out.mkdir(exist_ok=True)
info={};tiles=[]
for name in ['head.mp4','000000.mp4','right_wrist.mp4']:
 cap=cv2.VideoCapture(str(r/name));fps=cap.get(cv2.CAP_PROP_FPS);N=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));indices=set(np.linspace(0,max(N-1,0),7).astype(int));count=0
 while True:
  ok,im=cap.read()
  if not ok:break
  if count in indices:
   rgb=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);image=Image.fromarray(rgb);image.thumbnail((420,300));tile=Image.new('RGB',(440,330),(25,25,25));tile.paste(image,((440-image.width)//2,30));ImageDraw.Draw(tile).text((8,8),f'{name} f{count} {count/fps:.2f}s',fill='white');tiles.append(tile)
   Image.fromarray(rgb).save(out/f'{Path(name).stem}_{count:04d}.png')
  count+=1
 cap.release();info[name]={'frames_decoded':count,'declared_frames':N,'fps':fps,'sampled_indices':list(map(int,sorted(indices))),'full_decode_pass':count==N and count>0}
canvas=Image.new('RGB',(7*440,3*330),(25,25,25))
for i,t in enumerate(tiles):canvas.paste(t,((i%7)*440,(i//7)*330))
canvas.save(out/'contact_sheet.jpg',quality=92)
(out/'decode.json').write_text(json.dumps(info,indent=2));print(info)
