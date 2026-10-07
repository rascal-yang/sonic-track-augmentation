"""Compose original first-person footage, synchronized by simulation time."""
from pathlib import Path
import json
import subprocess

B = Path(__file__).resolve().parents[1]
rows = json.loads((B / 'summary.json').read_text())['trials']
assert len(rows) == 5
out = B / 'analysis/multistart_head_overview.mp4'
text_dir = B / 'analysis/head_overview_labels'
text_dir.mkdir(exist_ok=True)
font = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
labels = ['1  后方 30 cm', '2  后方 45 cm', '3  后方 70 cm',
          '4  左偏 25 cm · 后方 45 cm', '5  右偏 25 cm · 后方 45 cm']
inputs, durations, sources = [], [], []
for row in rows:
    source = B / 'runs' / row['run'] / 'head.mp4'
    probe = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate,nb_frames,duration',
        '-of', 'json', str(source)]))['streams'][0]
    assert (probe['width'], probe['height'], probe['r_frame_rate']) == (640, 480, '25/1')
    sources.append(dict(path=str(source), **probe))
    durations.append(float(probe['duration']))
    inputs += ['-i', str(source)]
duration = max(durations)
inputs += ['-f', 'lavfi', '-i', f'color=c=0x101820:s=640x540:r=25:d={duration}']

def draw(label, key, x, y, size=26, color='white', enable=None):
    p = text_dir / (key + '.txt')
    p.write_text(label)
    result = f'drawtext=fontfile={font}:textfile={p}:x={x}:y={y}:fontsize={size}:fontcolor={color}:expansion=none'
    if enable:
        result += f":enable='{enable}'"
    return result

filters = []
for i, row in enumerate(rows):
    end = durations[i]
    chain = (f'[{i}:v]setpts=PTS-STARTPTS,setsar=1,fps=25,'
             f'tpad=stop_mode=clone:stop_duration={duration},trim=duration={duration},'
             'pad=640:540:0:60:color=0x101820,')
    chain += draw(labels[i], f'group_{i}', 16, 18, 25)
    chain += ',' + draw('成功' if row['end_to_end_success'] else '失败',
                        f'result_{i}', 556, 18, 26,
                        '0x65DEA1' if row['end_to_end_success'] else '0xFF8585')
    if end < duration:
        chain += f",drawbox=x=0:y=505:w=640:h=35:color=black@0.65:t=fill:enable='gte(t,{end})'"
        chain += ',' + draw('本组已结束 · 保留末帧', f'end_{i}', 16, 510, 20,
                            enable=f'gte(t,{end})')
    filters.append(chain + f'[v{i}]')
legend = ['五组第一视角对比', 'SONIC 走近 → 侧向抓取', '4 / 5 成功 · 80%',
          '各起点一次，按仿真时间同步', '较早结束的组保留末帧', '同一场景、物体与初始朝向', '右偏组：接近时碰倒水瓶']
chain = '[5:v]setpts=PTS-STARTPTS'
for i, line in enumerate(legend):
    chain += ',' + draw(line, f'legend_{i}', 32, 48 + i*64,
                        34 if i == 0 else 26,
                        '0x65DEA1' if i == 2 else 'white')
filters.append(chain + '[v5]')
filters.append(''.join(f'[v{i}]' for i in range(6)) +
               'xstack=inputs=6:layout=0_0|640_0|1280_0|0_540|640_540|1280_540[out]')
subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                *inputs, '-filter_complex_threads', '2', '-filter_complex', ';'.join(filters),
                '-map', '[out]', '-an', '-c:v', 'libx264', '-threads', '4',
                '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out)], check=True)
subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(out), '-f', 'null', '-'], check=True)
probe = json.loads(subprocess.check_output([
    'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
    'stream=width,height,r_frame_rate,nb_frames,duration', '-of', 'json', str(out)]))['streams'][0]
assert (probe['width'], probe['height'], probe['r_frame_rate']) == (1920, 1080, '25/1')
assert abs(float(probe['duration']) - duration) < .041
(B / 'analysis/head_overview_validation.json').write_text(json.dumps({
    'output': str(out), 'stream': probe, 'full_decode_passed': True,
    'sources': sources, 'synchronization': 'simulation time from frame zero; no retiming',
    'layout': '3x2; groups1-3 top, groups4-5 bottom left/middle; legend bottom right',
    'padding': 'freeze final frame after each source ends, labeled on screen',
    'original_frame_resolution_preserved': True}, indent=2))
subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', '27', '-i', str(out),
                '-frames:v', '1', str(B/'analysis/head_overview_review.png')], check=True)
print(json.dumps(probe, indent=2))
print(out)
