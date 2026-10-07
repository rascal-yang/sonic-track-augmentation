"""Reduce duplicate PNG writes and cache compression cost; preserve all captured arrays."""
import json,zipfile
from pathlib import Path
import numpy as np

def install():
 import imageio.v2 as imageio
 original=imageio.imwrite;counts={}
 def write(uri,*args,**kwargs):
  name=Path(uri).name if isinstance(uri,(str,Path)) else ''
  if name in ['latest.png','latest_head.png','latest_right_wrist.png']:
   count=counts.get(name,0);counts[name]=count+1
   if count%25:return None
  return original(uri,*args,**kwargs)
 imageio.imwrite=write
 from training_capture import TrainingCapture
 def flush(self):
  if self.datasets:
   path=self.spool/f'chunk_{self.chunk_id:05d}.npz'
   with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as stream:
    for key,values in self.datasets.items():
     with stream.open(key+'.npy','w',force_zip64=True) as item:np.save(item,np.stack(values),allow_pickle=False)
   self.chunk_id+=1;self.datasets={}
  self.f.attrs['capture_efficiency']='All original frames/state arrays retained; lossless temporary NPZ deflate1; diagnostic latest PNG snapshots throttled25x, not video/RGB timestamps.'
  (self.spool/'attrs.json').write_text(json.dumps(self.f.attrs))
 TrainingCapture.flush_chunks=flush
 from capture_timing import install as install_timing
 install_timing(TrainingCapture)
