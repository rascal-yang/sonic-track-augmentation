from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py,numpy as np
B=Path(__file__).resolve().parents[1];summary=json.loads((B/'summary.json').read_text())
fig,axs=plt.subplots(1,2,figsize=(12,5));colors=plt.cm.tab10.colors
for idx,row in enumerate(summary['trials']):
 r=B/'runs'/row['run']
 with h5py.File(r/'episode_raw.h5') as f:
  root=f['observation/root_position_world'][:] if 'observation/root_position_world' in f else None
  if root is None:
   samples=[json.loads(l) for l in (r/'trajectory.jsonl').read_text().splitlines()];root=np.array([s['robot_root_pos'] for s in samples])
  t=f['next_observation/timestamp'][:] if 'next_observation/timestamp' in f else np.arange(len(root))*.02
  clearance=f['next_observation/metrics/whole_object_clearance_m'][:]*100
  axs[0].plot(root[:,0],root[:,1],label=row['run'],color=colors[idx]);axs[0].scatter(root[0,0],root[0,1],marker='o',color=colors[idx])
  axs[1].plot(np.arange(len(clearance))*.02,clearance,label=row['run'],color=colors[idx])
axs[0].scatter(-.09,-.54,marker='*',s=150,color='black',label='Standing goal');axs[0].plot([-.55,.55],[-.289,-.289],'k--',label='Desk front edge');axs[0].set(xlabel='World X (m)',ylabel='World Y (m)',title='Measured pelvis trajectories');axs[0].axis('equal');axs[0].legend(fontsize=7)
axs[1].axhline(2,ls='--',color='black',label='2cm threshold');axs[1].set(xlabel='Simulation time (s)',ylabel='Whole-bottle lowest-point clearance (cm)',title='Physical lift, all attempts');axs[1].legend(fontsize=7)
for ax in axs:ax.grid(alpha=.25)
fig.tight_layout();fig.savefig(B/'analysis/paths_and_clearance.png',dpi=160)
