"""Validate frame/action alignment, physical continuity, camera mounts and videos."""
import argparse,json,subprocess,hashlib
from pathlib import Path
import h5py,numpy as np
from scipy.spatial.transform import Rotation
p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);a=p.parse_args()
report={}
with h5py.File(a.run/'episode_raw.h5') as f:
    n=len(f['observation/timestamp']);dt=float(f.attrs['dt']);sub=int(round(dt/float(f.attrs['physics_dt'])));q=f['observation/joint_position'][:]
    assert f['action/joint_position_target'].shape==(n,43)
    assert np.array_equal(f['observation/control_index'][:],np.arange(n))
    assert np.allclose(f['observation/timestamp'][:],np.arange(n)*dt,atol=1e-10)
    assert len(f['physics/control_index'])==n*sub
    assert np.array_equal(f['physics/control_index'][:],np.repeat(np.arange(n),sub))
    assert np.array_equal(f['physics/substep_index'][:],np.tile(np.arange(sub),n))
    assert np.array_equal(f['physics/joint_position_target'][::sub],f['action/joint_position_target'][:])
    assert np.array_equal(f['physics/joint_position_target'][:],np.repeat(f['action/joint_position_target'][:],sub,axis=0))
    continuity=float(np.max(abs(f['next_observation/joint_position'][:-1]-q[1:])));assert continuity<1e-7
    checks={}
    for name in ['head','right_wrist']:
        idx=f[f'camera/{name}/control_index'][:];assert np.array_equal(idx,np.arange(0,n,2))
        assert np.allclose(f[f'camera/{name}/timestamp'][:],f['observation/timestamp'][idx],atol=1e-10)
        calib=json.loads(f.attrs['camera_calibration'])[name];body=json.loads(f.attrs['body_names']).index(calib['body_name'])
        qq=f['observation/body_quaternion_wxyz'][idx,body];position=f['observation/body_position_world'][idx,body]+Rotation.from_quat(qq[:,[1,2,3,0]]).apply(calib['body_local_translation_m'])
        mount_error=float(np.max(abs(position-f[f'camera/{name}/world_T_camera_cv'][:,:3,3])));assert mount_error<1e-6
        sample=np.unique(np.linspace(0,len(idx)-1,min(12,len(idx)),dtype=int));std=[float(np.std(f[f'camera/{name}/rgb'][i])) for i in sample];rgb_variance_ok=min(std)>4
        transforms=f[f'camera/{name}/world_T_camera_cv'][:];assert np.isfinite(transforms).all();assert np.allclose(np.linalg.det(transforms[:,:3,:3]),1,atol=1e-5)
        checks[name]={'frames':len(idx),'mount_max_error_m':mount_error,'sample_rgb_std_min':min(std),'rgb_variance_check_passed':rgb_variance_ok,'timestamps_match_pre_action_state':True}
    finite={}
    for prefix in ['observation','action','physics','next_observation']:
        def check(name,item):
            if isinstance(item,h5py.Dataset) and item.dtype.kind=='f':assert np.isfinite(item[:]).all(),item.name
        f[prefix].visititems(check);finite[prefix]=True
    strict=f['next_observation/metrics/strict_lift_now'][:].astype(bool);held=best=0;first=None
    for i,v in enumerate(strict):
        held=held+1 if v else 0;best=max(best,held)
        if held>=25 and first is None:first=(i+1)*dt
    depth=f['camera/head/distance_to_image_plane'];valid=f['camera/head/depth_valid'];depth_ok=[]
    for i in np.linspace(0,len(depth)-1,min(12,len(depth)),dtype=int):
        mask=valid[i];assert np.isfinite(depth[i]).all();assert np.all(depth[i][mask]>0);depth_ok.append(float(np.mean(mask)))
    report.update(control_steps=n,simulation_duration_s=n*dt,physics_substeps=n*sub,frame_action_alignment_passed=True,state_continuity_max_error=continuity,cameras=checks,finite_values=finite,depth_valid_fraction_min=min(depth_ok),success_ever=best>=25,success_final=held>=25,first_success_confirmation_s=first,strict_hold_max_s=best*dt,strict_hold_final_s=held*dt,max_whole_clearance_m=float(np.max(f['next_observation/metrics/whole_object_clearance_m'][:])),close_attempts=sum(json.loads(p.read_text())['action']=='close_hand' for p in a.run.glob('decision_[0-9][0-9][0-9].json')))
    assert np.linalg.norm(np.diff(f['camera/head/world_T_camera_cv'][:,:3,3],axis=0))>1e-4
    report['camera_follows_robot_verified']=True
videos={}
for name in ['head.mp4','right_wrist.mp4','000000.mp4']:
    path=a.run/name;v=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,width,height,nb_frames,r_frame_rate,duration','-of','json',str(path)]))['streams'][0]
    subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(path),'-f','null','-'],check=True)
    if name!='000000.mp4':assert int(v['nb_frames'])==len(range(0,n,2))
    videos[name]=dict(v,bytes=path.stat().st_size,full_decode_passed=True)
report['videos']=videos
(a.run/'validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
