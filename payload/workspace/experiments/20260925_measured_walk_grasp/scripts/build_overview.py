from pathlib import Path
import json,subprocess
B=Path(__file__).resolve().parents[1];protocol=json.loads((B/'protocol.json').read_text());names=protocol['evaluation_runs'];rows=[json.loads((B/'runs'/n/'summary.json').read_text()) for n in names];out=B/'reports/multistart_head.mp4';inputs=[];duration=0
for n in names:
 p=B/'runs'/n/'head.mp4';d=float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','default=nw=1:nk=1',str(p)]));duration=max(duration,d);inputs+=['-i',str(p)]
inputs+=['-f','lavfi','-i',f'color=c=0x12202c:s=640x540:r=25:d={duration}']
filters=[]
for i,(n,row) in enumerate(zip(names,rows)):
 label=f'{n}  '+('SUCCESS' if row['success'] else ('INTERRUPTED' if row.get('infrastructure_interrupted') else 'FAILED'))+f"  retries={row['retries']}"
 filters.append(f"[{i}:v]setpts=PTS-STARTPTS,setsar=1,tpad=stop_mode=clone:stop_duration={duration},trim=duration={duration},pad=640:540:0:60:color=0x12202c,drawtext=text='{label}':x=12:y=20:fontsize=18:fontcolor=white[v{i}]")
success=sum(x['success'] for x in rows);legend=["NEW SCENE / NEW MATERIALS",f'{success} of 5 starts succeeded','Live SONIC walking + arm IK','Slow contact / bounded retry','Videos at original simulation speed','Ended runs keep their last frame']
chain='[5:v]setpts=PTS-STARTPTS'
for i,l in enumerate(legend):chain+=f",drawtext=text='{l}':x=24:y={60+i*68}:fontsize=25:fontcolor=white"
filters.append(chain+'[v5]');filters.append(''.join(f'[v{i}]' for i in range(6))+'xstack=inputs=6:layout=0_0|640_0|1280_0|0_540|640_540|1280_540[out]')
subprocess.run(['ffmpeg','-v','error','-y',*inputs,'-filter_complex_threads','2','-filter_complex',';'.join(filters),'-map','[out]','-c:v','libx264','-threads','4','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(out)],check=True)
subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(out),'-f','null','-'],check=True);print(out)
