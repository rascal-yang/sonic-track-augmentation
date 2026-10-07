"""Repack newly collected HDF5 with gzip; verify every dataset and attribute first."""
import argparse,os
import h5py,numpy as np
from common import Path,read,write,sha


def repack(run):
    run=Path(run);report=run/'lossless_pack.json'
    if report.exists():return read(report)
    source=run/'episode_raw.h5';target=run/'episode_gzip_pending.h5'
    assert read(run/'collection_result.json')['validation_passed']
    assert not target.exists()
    before=source.stat().st_size;before_hash=sha(source);checked=0
    with h5py.File(source) as src,h5py.File(target,'w') as dst:
        for key,value in src.attrs.items():dst.attrs[key]=value
        def copy(name,item):
            if isinstance(item,h5py.Group):out=dst.require_group(name)
            elif item.ndim==0:out=dst.create_dataset(name,data=item[()])
            else:
                chunks=item.chunks or (min(max(len(item),1),64),*item.shape[1:])
                out=dst.create_dataset(name,shape=item.shape,dtype=item.dtype,maxshape=item.maxshape,
                    chunks=chunks,compression='gzip',compression_opts=4,shuffle=True)
                for start in range(0,len(item),16):out[start:start+16]=item[start:start+16]
            for key,value in item.attrs.items():out.attrs[key]=value
        src.visititems(copy)
    with h5py.File(source) as src,h5py.File(target) as dst:
        def verify(name,item):
            nonlocal checked
            other=dst[name]
            assert set(item.attrs)==set(other.attrs)
            for key in item.attrs:assert np.array_equal(item.attrs[key],other.attrs[key])
            if isinstance(item,h5py.Dataset):
                assert item.shape==other.shape and item.dtype==other.dtype
                if item.ndim==0:assert np.array_equal(item[()],other[()])
                else:
                    for start in range(0,len(item),16):
                        x=item[start:start+16];y=other[start:start+16]
                        assert x.tobytes()==y.tobytes(),name
                checked+=1
        src.visititems(verify)
        assert set(src.attrs)==set(dst.attrs)
        for key in src.attrs:assert np.array_equal(src.attrs[key],dst.attrs[key])
    result=dict(before_bytes=before,after_bytes=target.stat().st_size,source_sha256_before=before_hash,
                source_sha256_after=sha(target),verified_datasets=checked,all_dataset_bytes_identical=True,
                all_attributes_identical=True,RGB_depth_segmentation_preserved=True)
    if target.stat().st_size>=before:
        target.unlink();result['replaced']=False
    else:
        os.replace(target,source);result['replaced']=True
    write(report,result)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();print(repack(a.run))
