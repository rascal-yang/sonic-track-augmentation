"""Build a file://-compatible local sensor and telemetry viewer."""
import argparse
import json
from pathlib import Path

import h5py

parser = argparse.ArgumentParser()
parser.add_argument('--run', type=Path, required=True)
args = parser.parse_args()
base = Path(__file__).resolve().parents[1]
run = args.run.resolve()
with h5py.File(run / 'episode_raw.h5') as f:
    indices = f['camera/head/control_index'][:]
    data = {
        'joint_names': json.loads(f.attrs['joint_names']),
        'q': f['observation/joint_position'][indices].tolist(),
        'target': f['action/joint_position_target'][indices].tolist(),
        'clearance': f['observation/metrics/whole_object_clearance_m'][indices].tolist(),
        'contact': f['observation/metrics/object_hand_normal_force_N'][indices].tolist(),
        'root': f['observation/root_position'][indices].tolist(),
    }
html = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>G1 · 窗边房间视觉模仿数据</title>
<style>body{font:16px system-ui;background:#111923;color:#e4eaf3;max-width:1300px;margin:28px auto;padding:20px}h1{font-size:26px}p{line-height:1.7;color:#bac8db}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}video,img{width:100%;border-radius:10px;background:#000}section{background:#1b2838;padding:18px;border-radius:14px;margin:18px 0}a{color:#83c5ff}button,select{padding:8px;margin:6px;border-radius:6px}input{width:100%}pre{white-space:pre-wrap}canvas{width:100%;height:160px}small{color:#b3c3d7}</style>
<h1>G1 · 窗边房间视觉模仿数据</h1>
<p>4张手机照片约束的参数化重建；桌高按0.75m估计。不可见房间边界为推测。复用刚性玩偶。此条示范采用几何反馈监督器＋SONIC planner＋IK＋SONIC，未调用实时GPT，GRAIL残差关闭。</p>
<div class="grid"><section><h2>输入照片</h2><img src="inputs/room_03.jpg"></section><section><h2>三维重建</h2><img src="assets/preview_overview.png"></section></div>
<section><h2>同步传感器 · RUN_NAME</h2><div class="grid"><div>头部第一视角<video id="head" src="RUN_PATH/head.mp4" preload="metadata" playsinline></video></div><div>右腕视角（有自身遮挡）<video id="wrist" src="RUN_PATH/right_wrist.mp4" preload="metadata" muted playsinline></video></div></div>
<button id="play">播放 / 暂停</button><button id="prev">上一帧</button><button id="next">下一帧</button><span id="time"></span><input id="seek" type="range" min="0" max="749" value="0" step="1"><pre id="telemetry"></pre><label>关节 <select id="joint"></select></label><canvas id="chart" width="1100" height="160"></canvas><small>蓝色：实际角度；橙色：即将执行的目标。图像与本体状态均在动作前采样；接触与物体离桌高度属于特权信息，不进入LeRobot的observation.state。</small></section>
<section><h2>第三视角</h2><video controls src="RUN_PATH/000000.mp4" preload="metadata"></video><p>第三视角保留原记录器的起止附加帧，仅供观察，不用于训练帧对齐。</p></section>
<section><h2>数据与说明</h2><p><a href="README.md">运行说明</a> · <a href="REPORT.md">实验报告</a> · <a href="docs/DATA_SCHEMA.md">数据契约</a> · <a href="RUN_PATH/validation.json">自动验证结果</a> · <a href="assets/scene.usda">USD场景</a></p><p>训练包：datasets/g1_window_pickup（LeRobot v3）。原始数据：RUN_PATH/episode_raw.h5。</p></section>
<script>const D=DATA_JSON;const h=document.querySelector('#head'),w=document.querySelector('#wrist'),s=document.querySelector('#seek'),j=document.querySelector('#joint');D.joint_names.forEach((n,i)=>j.add(new Option(n,i)));s.max=D.q.length-1;
function draw(){let k=Math.min(D.q.length-1,Math.floor(h.currentTime*25+1e-4)),n=+j.value;s.value=k;document.querySelector('#time').textContent=`帧 ${k} / ${D.q.length-1} · ${(k/25).toFixed(2)} 秒`;document.querySelector('#telemetry').textContent=`整物最低碰撞点离桌 ${(D.clearance[k]*100).toFixed(2)} cm | 手部接触力 ${D.contact[k].toFixed(2)} N\n基座位置 [${D.root[k].map(x=>x.toFixed(3)).join(', ')}] m\n${D.joint_names[n]}: 实际 ${D.q[k][n].toFixed(4)} rad → 目标 ${D.target[k][n].toFixed(4)} rad`;
let c=document.querySelector('#chart'),ctx=c.getContext('2d'),a=D.q.map(x=>x[n]),b=D.target.map(x=>x[n]),lo=Math.min(...a,...b)-.05,hi=Math.max(...a,...b)+.05;ctx.clearRect(0,0,c.width,c.height);[a,b].forEach((v,z)=>{ctx.strokeStyle=['#70cfff','#ffbf70'][z];ctx.beginPath();v.forEach((v,i)=>{let x=i/(a.length-1)*c.width,y=c.height-10-(v-lo)/(hi-lo)*(c.height-20);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke()});ctx.strokeStyle='#fff';ctx.beginPath();ctx.moveTo(k/(a.length-1)*c.width,0);ctx.lineTo(k/(a.length-1)*c.width,c.height);ctx.stroke();}
function jump(k){h.pause();w.pause();h.currentTime=Math.max(0,Math.min(D.q.length-1,k))/25;w.currentTime=h.currentTime;draw()}
document.querySelector('#play').onclick=()=>{if(h.paused){w.currentTime=h.currentTime;h.play();w.play()}else{h.pause();w.pause()}};s.oninput=()=>jump(+s.value);document.querySelector('#prev').onclick=()=>jump(+s.value-1);document.querySelector('#next').onclick=()=>jump(+s.value+1);j.onchange=draw;h.ontimeupdate=()=>{if(Math.abs(w.currentTime-h.currentTime)>.06)w.currentTime=h.currentTime;draw()};h.onended=()=>w.pause();draw();</script></html>'''
html = html.replace('RUN_NAME', run.name).replace('RUN_PATH', str(run.relative_to(base)))
html = html.replace('DATA_JSON', json.dumps(data, separators=(',', ':')))
(base / 'index.html').write_text(html)
print(base / 'index.html')
