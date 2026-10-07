from pathlib import Path
import json,subprocess
import h5py,numpy as np
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parents[1];allrows=[];fig,axs=plt.subplots(2,2,figsize=(11,7))
for name,label in [('stationary_002','Baseline'),('stationary_004','Guarded optimized')]:
 r=B/'runs'/name;s=json.loads((r/'summary.json').read_text());ds=[json.loads(p.read_text()) for p in sorted(r.glob('decision_[0-9][0-9][0-9].json'))];obs={json.loads(p.read_text())['observation_id']:json.loads(p.read_text()) for p in r.glob('observation_[0-9][0-9][0-9].json')};close=next(obs[d['observation_id']]['step'] for d in ds if d['action']=='close_hand');first=next(obs[d['observation_id']]['step'] for d in ds if d['action']=='reach')
 with h5py.File(r/'episode_raw.h5') as f:
  bn=json.loads(f.attrs['body_names']);pos=f['observation/body_position_world'][:,bn.index('right_wrist_yaw_link')];t=f['observation/timestamp'][:];clear=f['next_observation/metrics/whole_object_clearance_m'][:];vel=savgol_filter(pos,11,3,deriv=1,delta=.02,axis=0);acc=savgol_filter(pos,11,3,deriv=2,delta=.02,axis=0);jerk=savgol_filter(pos,11,3,deriv=3,delta=.02,axis=0);idx=slice(close-100,close)
  row={'run':name,'duration_s':s['duration_s'],'first_success_confirmation_s':s['first_lift_confirmation_s'],'preclosure_goals':s['preclosure_waypoint_count'],'contact_window_seconds_before_closure':2,'contact_window_wrist_speed_p95_m_s':float(np.percentile(np.linalg.norm(vel[idx],axis=1),95)),'contact_window_wrist_acceleration_rms_m_s2':float(np.sqrt(np.mean(np.sum(acc[idx]**2,axis=1)))),'contact_window_wrist_jerk_rms_m_s3':float(np.sqrt(np.mean(np.sum(jerk[idx]**2,axis=1)))),'derivative_method':'Savitzky-Golay 11 samples=.22s, degree3, same filter for both runs; actual simulated wrist positions'};allrows.append(row)
  axs[0,0].plot(pos[first:close,1],pos[first:close,2],label=label);axs[0,1].plot(t,clear*100,label=label);axs[1,0].plot(t[close-100:close]-t[close],np.linalg.norm(vel[idx],axis=1),label=label);axs[1,1].plot(t[close-100:close]-t[close],np.linalg.norm(jerk[idx],axis=1),label=label)
axs[0,0].set(xlabel='Wrist world Y (m)',ylabel='Wrist world Z (m)',title='Approach path');axs[0,1].set(xlabel='Simulation time (s)',ylabel='Whole bottle clearance (cm)',title='Physical lift');axs[1,0].set(xlabel='Seconds before closure',ylabel='Wrist speed (m/s)',title='Actual contact-approach speed');axs[1,1].set(xlabel='Seconds before closure',ylabel='Filtered jerk (m/s^3)',title='Actual contact-approach jerk')
for ax in axs.flat:ax.legend();ax.grid(alpha=.25)
fig.tight_layout();fig.savefig(B/'reports/stationary_comparison.png',dpi=160);plt.close(fig);(B/'reports/stationary_comparison.json').write_text(json.dumps(allrows,indent=2));print(json.dumps(allrows,indent=2))
out=B/'reports/stationary_before_after.mp4';duration=max(r['duration_s'] for r in allrows)
inputs=[];filters=[]
for i,name in enumerate(['stationary_002','stationary_004']):
 inputs+=['-i',str(B/'runs'/name/'head.mp4')];label='Baseline 19.22 s' if i==0 else 'Guarded optimized 17.42 s';filters.append(f"[{i}:v]setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration={duration},trim=duration={duration},pad=640:520:0:40:color=0x12202c,drawtext=text='{label}':x=20:y=10:fontsize=22:fontcolor=white[v{i}]")
filters.append('[v0][v1]hstack=inputs=2[out]');subprocess.run(['ffmpeg','-v','error','-y',*inputs,'-filter_complex_threads','2','-filter_complex',';'.join(filters),'-map','[out]','-c:v','libx264','-threads','2','-crf','20','-pix_fmt','yuv420p',str(out)],check=True);subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(out),'-f','null','-'],check=True)
