"""Replay this session's inspected, object-free approach prefix with feedback guards."""
import json,time,sys
from pathlib import Path
r=Path(sys.argv[1]);source=r.parent/'trial_001'
for idx in range(8):
 f=r/f'observation_{idx:03}.json'
 while not f.exists():time.sleep(1)
 s=json.loads(f.read_text());w=s.get('contact_watch',{});m=s['grasp_information']['metrics']
 assert not w.get('alert') and w.get('peak_total_table_force_N',0)<.05,s
 assert s['robot_root_pos'][2]>.65 and m['object_hand_normal_force_N']<.1,s
 assert abs(m['whole_object_clearance_m'])<.003,s
 d=json.loads((source/f'decision_{idx:03}.json').read_text())
 if idx==3:assert s['navigation_error_m']<.06,s
 if idx==4:assert s['navigation_error_m']<.07,s
 if idx==6:assert s['phase']=='grasp' and s['grail_phase_gain']==1,s
 d['author']='Replay of current-session inspected approach prefix with actual-state guards; no new GPT inference for this checkpoint'
 d['reason']='Replay trial_001 approach: '+d['reason'];d['wall_time']=time.time()
 p=r/f'decision_{idx:03}.json';assert not p.exists();p.with_suffix('.tmp').write_text(json.dumps(d,indent=2));p.with_suffix('.tmp').replace(p)
 print('replayed',idx,'step',s['step'],'root',s['robot_root_pos'],'navigation_error',s.get('navigation_error_m'),flush=True)
print('Prefix complete; return to live GPT feedback at observation 008.',flush=True)
