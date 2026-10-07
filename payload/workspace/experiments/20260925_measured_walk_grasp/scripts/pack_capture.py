"""Build HDF5 outside Isaac/Kit, preserving append order and original schema."""
from pathlib import Path
import sys,json,numpy as np,h5py
r=Path(sys.argv[1]);out=r/'episode_raw.h5'
assert not out.exists(),out
with h5py.File(out,'w') as f:
 for k,v in json.loads((r/'capture_chunks/attrs.json').read_text()).items():f.attrs[k]=v
 for chunk in sorted((r/'capture_chunks').glob('chunk_*.npz')):
  with np.load(chunk) as z:
   for k in z.files:
    a=z[k]
    if k not in f:
     opts={'compression':'lzf','shuffle':True} if a.ndim>=3 and a[0].size>1000 else {}
     f.create_dataset(k,data=a,maxshape=(None,*a.shape[1:]),chunks=(1,*a.shape[1:]),**opts)
    else:
     d=f[k];n=len(d);d.resize(n+len(a),axis=0);d[n:]=a
print(out)
