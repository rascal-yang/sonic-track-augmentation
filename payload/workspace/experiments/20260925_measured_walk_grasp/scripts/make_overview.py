from pathlib import Path
import json,subprocess
B=Path(__file__).resolve().parents[1];rows=json.loads((B/'summary.json').read_text())['trials']
labels=['Behind 30 cm','Behind 45 cm','Behind 70 cm','Left 25 / behind 45 cm','Right 25 / behind 45 cm']
inputs=[];durations=[]
for row in rows:
 p=B/'runs'/row['run']/'000000.mp4';inputs.extend(['-i',str(p)])
 durations.append(float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','csv=p=0',str(p)])))
filters=[]
for i,row in enumerate(rows):
 label=labels[i]+' | '+('SUCCESS' if row['end_to_end_success'] else 'FAIL')
 filters.append(f'[{i}:v]scale=480:272,setsar=1,fps=25,tpad=stop_mode=clone:stop_duration=60,trim=duration={max(durations)},setpts=PTS-STARTPTS,drawtext=text=\'{label}\':x=8:y=8:fontsize=17:fontcolor=white:box=1:boxcolor=black@0.6[v{i}]')
filters.append(''.join(f'[v{i}]' for i in range(len(rows)))+f'xstack=inputs={len(rows)}:layout=0_0|480_0|960_0|0_272|480_272:fill=black[out]')
out=B/'analysis/multistart_overview.mp4'
subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y',*inputs,'-filter_complex',';'.join(filters),'-map','[out]','-an','-c:v','libx264','-threads','4','-crf','23','-pix_fmt','yuv420p',str(out)],check=True)
subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(out),'-f','null','-'],check=True)
print(out)
