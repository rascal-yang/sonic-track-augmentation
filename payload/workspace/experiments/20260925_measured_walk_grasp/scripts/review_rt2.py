from pathlib import Path
import json
import numpy as np,h5py,cv2
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parents[1];r=B/'renders/walk_010_arc_rt2';raw=B/'runs/walk_010_arc/episode_raw.h5'
with h5py.File(r/'frames.h5') as f,h5py.File(raw) as src:
 print('frame keys',list(f));idx=f['control_index'][:];assert np.array_equal(idx,src['camera/head/control_index'][:]);assert np.allclose(f['timestamp'][:],src['camera/head/timestamp'][:]);err=float(np.abs(f['world_T_camera_cv'][:]-src['camera/head/world_T_camera_cv'][:]).max());assert err<1e-6;N=len(idx)
cap=cv2.VideoCapture(str(r/'video.mp4'));fps=cap.get(cv2.CAP_PROP_FPS);chosen=[0,200,415,465,575,645,675,720,762,N-1];n=0;tiles=[]
while True:
 ok,im=cap.read()
 if not ok:break
 if n in chosen:
  rgb=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);im=Image.fromarray(rgb);im.save(r/f'review_{n:04d}.png');im.thumbnail((320,240));tile=Image.new('RGB',(330,265),(25,25,25));tile.paste(im,(5,25));ImageDraw.Draw(tile).text((5,5),f'RT2 frame{n} t={n/fps:.2f}s',fill='white');tiles.append(tile)
 n+=1
cap.release();assert n==N and fps==25
canvas=Image.new('RGB',(330*5,265*2),(25,25,25))
for i,im in enumerate(tiles):canvas.paste(im,((i%5)*330,(i//5)*265))
canvas.save(r/'review_sheet.jpg',quality=94)
report={'frames_decoded':n,'fps':fps,'full_decode_passed':True,'indices_match_original':True,'timestamps_match_original':True,'camera_matrix_max_error':err,'sampled_indices':chosen,'physics_advanced':False};(r/'validation.json').write_text(json.dumps(report,indent=2));print(report)
