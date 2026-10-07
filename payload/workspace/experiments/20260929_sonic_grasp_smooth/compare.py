"""Compare same-condition executed trajectories and plot evidence."""
from pathlib import Path
import argparse,json
import numpy as np,h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter
p=argparse.ArgumentParser();p.add_argument('baseline',type=Path);p.add_argument('candidate',type=Path);a=p.parse_args();out=a.candidate.parent.parent
b=json.loads((a.baseline/'quality.json').read_text());c=json.loads((a.candidate/'quality.json').read_text())
metrics={}
for k in b['motion_metrics']:
 if isinstance(b['motion_metrics'][k],(int,float)):
  bv=b['motion_metrics'][k];cv=c['motion_metrics'][k];metrics[k]={'baseline':bv,'candidate':cv,'change_percent':100*(cv/bv-1) if bv else None}
comparison=dict(baseline=b['run'],candidate=c['run'],baseline_success=b['task_success'],candidate_success=c['task_success'],duration_baseline_s=b['duration_s'],duration_candidate_s=c['duration_s'],duration_reduction_s=b['duration_s']-c['duration_s'],duration_reduction_percent=100*(1-c['duration_s']/b['duration_s']),motion_metrics=metrics,notes=['One selected deterministic trial per version; not a perturbation success rate.','Compare same analysis filter; active intervals end at each nominal lift completion.','Multiple reference changes together: simplification, interpolation, retiming and recomputed velocities; no isolated causal attribution.'])
(out/'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
fig,axes=plt.subplots(3,2,figsize=(12,10))
for r,label in [(a.baseline,'25mm original'),(a.candidate,'25mm simplified')]:
 with h5py.File(r/'episode_raw.h5') as f:
  n=len(f['observation/timestamp']);t=np.arange(n)*.02;idx=json.loads(f.attrs['body_names']).index('right_wrist_yaw_link');pos=f['next_observation/body_position_world'][:,idx];vel=savgol_filter(pos,11,3,deriv=1,delta=.02,axis=0);acc=savgol_filter(pos,11,3,deriv=2,delta=.02,axis=0)
  axes[0,0].plot(t,f['next_observation/metrics/whole_object_clearance_m'][:]*100,label=label)
  axes[0,1].plot(t,np.linalg.norm(vel,axis=1),label=label)
  axes[1,0].plot(t,np.linalg.norm(acc,axis=1),label=label)
  wr=R.from_quat(f['next_observation/body_quaternion_wxyz'][:,idx][:,[1,2,3,0]]);ob=R.from_quat(f['next_observation/object_quaternion_wxyz'][:][:,[1,2,3,0]])
  relative=wr.inv().apply(f['next_observation/object_position'][:]-pos);last=relative[-150:];axes[1,1].plot(np.arange(len(last))*.02,np.linalg.norm(last-last[0],axis=1)*1000,label=label)
  axes[2,0].plot(t,np.rad2deg(np.arccos(np.clip(ob.apply(np.tile([0,0,1],(n,1)))[:,2],-1,1))),label=label)
  names=json.loads(f.attrs['joint_names']);ids=[j for j,x in enumerate(names) if x.startswith('right_') and any(k in x for k in ['shoulder','elbow','wrist'])];target=f['action/joint_position_target'][:,ids]
  axes[2,1].plot(t[1:],np.max(abs(np.diff(target,axis=0)),axis=1),label=label)
for ax,title,unit in zip(axes.flat,['Whole bottle clearance','Actual wrist speed','Actual wrist acceleration','Object displacement in wrist: final3s','Bottle absolute tilt','Largest right-arm target increment'],['cm','m/s','m/s2','mm','deg','rad/20ms']):
 ax.set_title(title);ax.set_xlabel('Time(s)');ax.set_ylabel(unit);ax.grid(alpha=.3);ax.legend()
fig.suptitle('Fixed SONIC tokens + live decoder; identical scene and acceptance');fig.tight_layout();fig.savefig(out/'comparison.png',dpi=150)
print(json.dumps(comparison,indent=2))
