"""Classify a physical reference rollout without generic controller labels."""
from pathlib import Path
import argparse,json
import numpy as np
import h5py
p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();r=a.run
ref=json.loads((r/'reference_rollout.json').read_text())
with np.load(ref['reference'],allow_pickle=False) as z,h5py.File(r/'episode_raw.h5','r+') as f:
 n=len(f['observation/timestamp']);m=f['next_observation/metrics'];strict=m['strict_lift_now'][:];hold=0
 for value in strict[::-1]:
  if not value:break
  hold+=1
 success=hold*.02>=3 and bool(m['opposed_digits_contact'][-1])
 names=json.loads(f.attrs['joint_names']);ids=[names.index(str(name)) for name in z['g1_joint_names']]
 idx=np.minimum(np.arange(n),len(z['time_s'])-1)
 root_error=np.linalg.norm(f['observation/root_position'][:]-z['root_position_world_m'][idx],axis=1)
 joint_error=f['observation/joint_position'][:,ids]-z['g1_joint_position_rad'][idx]
 bn=json.loads(f.attrs['body_names']);zb=z['body_names'].tolist();j=bn.index('right_wrist_yaw_link');k=zb.index('right_wrist_yaw_link')
 wrist_error=np.linalg.norm(f['observation/body_position_world'][:,j]-z['body_position_world_m'][idx,k],axis=1)
 contacts=[json.loads(x) for x in (r/'table_contacts_physics.jsonl').read_text().splitlines()]
 peak=max((x['total_table_force_N'] for x in contacts),default=0.)
 integrity=[json.loads(x) for x in (r/'reference_integrity.jsonl').read_text().splitlines()]
 assert len(integrity)==n
 assert all(x['source_index']==min(i,len(z['time_s'])-1) and x['frame']==i and x['joint_reference_max_error_rad']<1e-6 for i,x in enumerate(integrity))
 decisions=[json.loads(x.read_text()) for x in sorted(r.glob('decision_[0-9][0-9][0-9].json'))]
 result=json.loads((r/'result.json').read_text())
 summary={'run':r.name,'test':'recorded actual body states as SONIC reference + recorded Revo2 position targets; no live arm IK/navigation', 'success':bool(success),'control_steps':n,'duration_s':n*.02,'source_samples':len(z['time_s']),'whole_reference_completed':n>=len(z['time_s']), 'strict_final_hold_s':hold*.02,'final_whole_bottle_clearance_m':float(m['whole_object_clearance_m'][-1]),'final_opposed_contact':bool(m['opposed_digits_contact'][-1]),'obstacle_contact_peak_N':peak,'minimum_pelvis_z_m':float(f['next_observation/root_position'][:,2].min()),'root_tracking_error_final_m':float(root_error[-1]),'root_tracking_error_max_m':float(root_error.max()),'wrist_tracking_error_final_m':float(wrist_error[-1]),'wrist_tracking_error_max_m':float(wrist_error.max()),'body_joint_error_rmse_rad':float(np.sqrt(np.mean(joint_error**2))),'reference_integrity_all_steps_passed':True,'stop_reason':decisions[-1]['reason'],'runtime_error':result.get('error'),'physical_state_replay':False,'training_ready':False}
 f.attrs['controller']=summary['test'];f.attrs['episode_success']=bool(success);f.attrs['training_ready']=False;f.attrs['reference_rollout']=json.dumps(ref)
 np.savez_compressed(r/'tracking_errors.npz',time_s=f['observation/timestamp'][:],root_error_m=root_error,wrist_error_m=wrist_error,body_joint_error_rad=joint_error)
(r/'rollout_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(summary,indent=2,ensure_ascii=False))
