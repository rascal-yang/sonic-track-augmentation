"""Report preserved partial walk005 without manufacturing a completed episode."""
from pathlib import Path
import json,time
B=Path(__file__).resolve().parents[1];r=B/'runs/walk_005'
s=json.loads((r/'observation_072.json').read_text())
cs=[json.loads(x) for x in (r/'table_contacts_physics.jsonl').read_text().splitlines()]
reason='Supervisor numpy.bool_ serialization error at preclose gate; launcher exited and simulator was no longer alive. No closure; no completed HDF5/result.'
summary={'run':r.name,'success':False,'clean_success':False,'control_steps':1335,'duration_s':26.7,'duration_basis':'last observation; interrupted prefix only','strict_final_hold_s':0.,'final_whole_bottle_clearance_m':s['grasp_information']['metrics']['whole_object_clearance_m'],'table_contact_peak_N':max(sum(v for k,v in c['links_N'].items() if not k.startswith('room:')) for c in cs),'room_contact_peak_N':max(sum(v for k,v in c['links_N'].items() if k.startswith('room:')) for c in cs),'contact_log_coverage_s':cs[-1]['time_s'],'navigation_reached_grasp':True,'attempts':0,'stop_reason':reason,'runtime_result':{'error':reason,'completed_result_available':False},'raw_data_complete':False,'training_ready':False,'unexecuted_decisions':['decision_072.json'],'stability_correction_physically_verified':False}
(r/'summary.json').write_text(json.dumps(summary,indent=2))
(r/'interruption.json').write_text(json.dumps({'time':time.time(),'reason':reason,'resume_requested':True,'resume_executed':False,'last_observation_step':1335,'last_contact_control_index':cs[-1]['control_index'],'no_simulator_process_confirmed':True,'source_correction':'scripts/optimized_protocol.py casts bool; scripts/launch_optimized.py requests finalized stop on future supervisor error','verification':'Python compilation and recorded-state predicate check pass; future graceful error handling not physically retested'},indent=2))
review={'accepted':False,'training_ready':False,'reviewed':['visual_review/contact_sheet.jpg','observation_000.png','observation_061.png','observation_061_head.png','observation_063.png','observation_063_head.png','observation_072_head.png'],'full_decode':json.loads((r/'visual_review/decode.json').read_text()),'spatial_findings':['Both hands visible in external sampled views; initial hands open behind desk','Upright bottle at last observation, open hand straddles bottle; no executed closure','No obvious gross table penetration in sampled views; last15 control steps missing flushed contact rows'],'temporal_review_status':'not certified; sampled-frame review and full decode do not establish native-rate temporal stability','data_limitation':'No finalized episode_raw.h5, result.json, or final next_observation/substep coverage; not training eligible'}
(r/'visual_realism_review.json').write_text(json.dumps(review,indent=2))
(r/'REPORT.md').write_text('''# walk_005：闭指前程序中断

走近、站立交接和抬手完成，最后观测在第1335步（26.70秒）。未闭指，未拿起；**不是成功抓取，也不是完成的稳定等待测试**。

新增闭指门限在首个检查点计算出：最近0.5秒手掌峰值速度0.1393 m/s、腕角速度0.5742 rad/s、位置误差2.887 cm，速度尚未收敛。随后日志序列化numpy.bool_报错，启动器结束，仿真进程丢失。修复并尝试恢复后确认无法继续；decision_072.json未执行。未重置并伪装成同一段连续实验。

保留至第1335步的轨迹和图像、26.40秒的物理接触日志（已保存部分桌/房间接触峰值均为0）。缺失最后15个控制步的完整接触行及最终HDF5，因此不能通过完整数据对齐或最终子步验收。三个视频均可完整解码，但仅进行抽帧和关键状态检查，视觉时序未认证，training_ready=false。

已修复布尔值序列化，并让未来启动器在监督程序错误时请求收尾保存而不是立即退出。编译和已记录状态的门限计算检查通过；修复后的完整抓取和异常收尾尚未重新进行物理验证。

[第一视角](head.mp4) · [第三视角](000000.mp4) · [腕部视角](right_wrist.mp4) · [部分结果](summary.json) · [中断详情](interruption.json)
''')
print('Interrupted prefix reported',summary['duration_s'])
