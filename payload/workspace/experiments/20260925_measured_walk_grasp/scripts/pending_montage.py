from pathlib import Path
import json
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parents[1];pending=[]
for p in sorted((B/'runs').glob('*/review_request_*.json')):
 if (p.parent/'result.json').exists() or p.with_name(p.name.replace('review_request_','reviewed_')).exists():continue
 pending.append((p,json.loads(p.read_text())))
if not pending:print('No pending visual gates');raise SystemExit
canvas=Image.new('RGB',(1280,400*len(pending)),(18,25,33));draw=ImageDraw.Draw(canvas)
for row,(p,q) in enumerate(pending):
 draw.text((12,row*400+8),p.parent.name+'  '+str(q['observation_id'])+'  '+q['label'],fill='white')
 for col,key in enumerate(['image','head']):
  f=Path(q[key])
  if f.exists():
   im=Image.open(f).convert('RGB');im.thumbnail((630,350));canvas.paste(im,(col*640+(640-im.width)//2,row*400+40))
out=B/'reports/pending_reviews.jpg';canvas.save(out,quality=94);print(out)
print([(p.parent.name,q['observation_id']) for p,q in pending])
