"""Compact only this new audited episode; preserve physical/action evidence and videos."""
from pathlib import Path
import shutil
import h5py,numpy as np
from common import read,write

def compact(run,keep_raw=False):
 r=Path(run);record=read(r/'collection_result.json');assert record['validation_passed']
 source=r/'episode_raw.h5';target=r/'episode_state.h5';removed=[]
 if not keep_raw and source.exists():
  if target.exists():raise RuntimeError('Refuse to overwrite compact state file')
  with h5py.File(source) as src,h5py.File(target,'w') as dst:
   for k,v in src.attrs.items():dst.attrs[k]=v
   dst.attrs['camera_pixels']='External head.mp4 and right_wrist.mp4; video is lossy preview, original RGB/depth/segmentation omitted after full capture audits.'
   def copy(name,item):
    if isinstance(item,h5py.Group):dst.require_group(name)
    elif isinstance(item,h5py.Dataset):
     if name.startswith('camera/') and name.rsplit('/',1)[-1] in ['rgb','distance_to_image_plane','depth_valid','semantic_segmentation']:return
     dst.create_dataset(name,data=item[:],compression='gzip')
   src.visititems(copy)
   def check(name,item):
    if isinstance(item,h5py.Dataset):assert np.array_equal(item[:],src[name][:]),name
   dst.visititems(check)
  removed.append(dict(path=source.name,bytes=source.stat().st_size,reason='Verified identical non-pixel state/action arrays retained in episode_state.h5; camera MP4 retained'))
  source.unlink()
 for name in ['capture_chunks','isaac_initial_scene.usda','synthetic','hdf5']:
  p=r/name
  if not p.exists():continue
  size=sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) if p.is_dir() else p.stat().st_size
  removed.append(dict(path=name,bytes=size,reason='Generated cache/scene duplicate; references, source snapshots, metadata and videos retained'))
  if p.is_dir():shutil.rmtree(p)
  else:p.unlink()
 # Intermediate checkpoint screenshots duplicate the retained videos; retain endpoints and review montage.
 snapshots=sorted(r.glob('observation_*.png'))
 keep=set(snapshots[:2]+snapshots[-2:])
 for p in snapshots:
  if p in keep:continue
  removed.append(dict(path=p.name,bytes=p.stat().st_size,reason='Intermediate screenshot represented in retained videos'));p.unlink()
 write(r/'cleanup_manifest.json',dict(scope='New generated episode only',removed=removed,freed_bytes=sum(x['bytes'] for x in removed),raw_pixels_retained=keep_raw,state_arrays_verified_identical=True))
 return sum(x['bytes'] for x in removed)
