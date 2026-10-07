from pathlib import Path
import csv,json
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parents[1]
for run in (B/'runs').iterdir():
 if not run.is_dir() or not (run/'part_distances.csv').exists():continue
 target='near_leg' if run.name.startswith('leg') else 'near_forearm'
 rows=[r for r in csv.DictReader((run/'part_distances.csv').open()) if r['part']==target];t=[float(r['time_s']) for r in rows]
 f,axs=plt.subplots(2,1,figsize=(10,6),sharex=True)
 for key,label in [('palm_to_part_center_m','IK palm to part center'),('pinch_to_part_center_m','Pinch center to part center')]:axs[0].plot(t,[100*float(r[key]) for r in rows],'-o',label=label)
 axs[0].set_ylabel('Distance (cm)');axs[0].set_title(run.name+' / '+target+' / measured feedback checkpoints');axs[0].legend()
 for key,label in [('thumb_force_N','Thumb'),('index_middle_force_N','Index + middle'),('table_force_N','Table support')]:axs[1].plot(t,[float(r[key]) for r in rows],'-o',label=label)
 axs[1].set_ylabel('Contact normal force (N)');axs[1].set_xlabel('Simulation time (s)');axs[1].legend()
 for ax in axs:ax.grid(alpha=.3)
 f.tight_layout();f.savefig(run/'feedback_curves.png',dpi=140);plt.close(f)
