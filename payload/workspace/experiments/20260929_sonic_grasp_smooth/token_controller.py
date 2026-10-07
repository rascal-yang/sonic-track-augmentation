"""Release ONNX decoder drives all29 G1 joints; no IK/planner actions."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
from scipy.spatial.transform import Rotation
from collections import deque

def install(token_file, decoder_file, parameter_file, ort_path):
    sys.path.insert(0,str(ort_path))
    import onnxruntime as ort
    import walk_grasp,brainco_adapter
    from direct_bottle import DirectBottlePlanner
    from experiment import write_json
    z=np.load(token_file,allow_pickle=False);tokens=z['sonic_encoded_tokens'].copy()
    params=json.loads(Path(parameter_file).read_text());order=np.array(params['mujoco_to_isaaclab'])
    default=np.array(params['default_angles'])[order];scale=np.array(params['action_scale'])[order]
    kp=np.array(params['kp'])[order];kd=np.array(params['kd'])[order]
    names=z['encoder_joint_names'].tolist()
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    net=ort.InferenceSession(str(decoder_file),sess_options=options,providers=['CPUExecutionProvider'])
    assert net.get_inputs()[0].shape==[1,994] and net.get_outputs()[0].shape==[1,29]
    class TokenPlanner(DirectBottlePlanner):
        def __init__(self,env,out,sonic):
            super().__init__(env,out,sonic);self.phase='token_rollout';self.goal=None
        def tick(self):pass
        def apply(self,d):
            assert d['action'] in ('continue','stop')
    walk_grasp.WalkGraspPlanner=TokenPlanner
    original=brainco_adapter.install_extension
    def extension_install(ext):
        original(ext);ext.controller='sonic_full'
        from initialization import install_restore
        install_restore(ext, Path(token_file).parent/'initial_state.npz')
        import shutil
        snapshot=ext.out/'token_source_snapshot';snapshot.mkdir(exist_ok=True)
        for file in Path(__file__).parent.glob('*.py'):shutil.copy2(file,snapshot/file.name)
        for name in ['plan.json','deployment_parameters.json']:shutil.copy2(Path(token_file).parent/name,snapshot/name)
        import torch
        robot=ext.env.scene['robot'];ids=[robot.joint_names.index(n) for n in names]
        state={'step':0,'target':None,'previous':np.zeros(29),'configured':False}
        history=deque(maxlen=10);records=[]
        old_before=ext.before_step
        def tensor(x):return torch.as_tensor(x,dtype=torch.float32,device=ext.env.device)[None]
        # Old ATM remains in the collector's plumbing, but cannot contribute any body action.
        ext.wrapped.action_transform_module.forward=lambda *a,**k: torch.zeros((1,29),device=ext.env.device)
        term=ext.env.action_manager.get_term('joint_pos');old_apply=term.apply_actions
        def apply_actions():
            old_apply()
            if state['target'] is not None:
                robot.set_joint_position_target(tensor(state['target']),joint_ids=ids)
                robot.set_joint_velocity_target(tensor(np.zeros(29)),joint_ids=ids)
                robot.set_joint_effort_target(tensor(np.zeros(29)),joint_ids=ids)
        term.apply_actions=apply_actions
        def before():
            old_before()
            if not state['configured']:
                robot.write_joint_stiffness_to_sim(tensor(kp),joint_ids=ids)
                robot.write_joint_damping_to_sim(tensor(kd),joint_ids=ids)
                write_json(ext.out/'token_actuator_setup.json',{'joint_names':names,'joint_ids':ids,'kp':kp.tolist(),'kd':kd.tolist(),'default':default.tolist(),'action_scale':scale.tolist(),'effort_limits_unchanged':True})
                state['configured']=True
            i=state['step'];assert i<len(tokens)
            d=robot.data;r=Rotation.from_quat(d.root_quat_w[0].cpu().numpy()[[1,2,3,0]])
            row={'q':d.joint_pos[0,ids].cpu().numpy().astype(float)-default,'dq':d.joint_vel[0,ids].cpu().numpy().astype(float),'omega':r.inv().apply(d.root_ang_vel_w[0].cpu().numpy()),'gravity':r.inv().apply([0,0,-1]),'last':state['previous'].copy()}
            if not history:
                for _ in range(9):history.append(row)
            history.append(row)
            obs=np.concatenate([tokens[i]]+[np.concatenate([h[k] for h in history]) for k in ['omega','q','dq','last','gravity']]).astype(np.float32)
            assert obs.shape==(994,) and np.isfinite(obs).all()
            raw=net.run(None,{net.get_inputs()[0].name:obs[None]})[0][0]
            assert np.isfinite(raw).all()
            target=default+scale*raw
            state['target']=target;state['previous']=raw.copy()
            for side in ['left','right']:
                hand=ext.env.scene['brainco_'+side];hn=z[f'revo2_{side}_active_joint_names'].tolist();hi=[hand.joint_names.index(n) for n in hn]
                hand.set_joint_position_target(tensor(z[f'revo2_{side}_active_position_target_rad'][i]),joint_ids=hi)
            records.append({'input':obs,'output':raw.copy(),'target':target.copy()})
            # Frequent lossless chunks survive a failure; no large JSON numerical logs.
            if i%50==49 or i==len(tokens)-1:
                np.savez_compressed(ext.out/f'decoder_chunk_{i//50:04d}.npz',control_index=np.arange(i-len(records)+1,i+1),inputs=np.stack([x['input'] for x in records]),outputs=np.stack([x['output'] for x in records]),targets=np.stack([x['target'] for x in records]))
                records.clear()
            with (ext.out/'token_progress.jsonl').open('a') as f:f.write(json.dumps({'step':i,'reference_token_index':i,'source_control_coordinate':float(z['source_control_coordinate'][i]) if 'source_control_coordinate' in z else i+830,'target_max_abs_rad':float(abs(target).max()),'body_IK':False})+'\n')
            state['step']+=1
        ext.before_step=before
        # Preserve final short decoder chunk during collector finalization.
        from training_capture import TrainingCapture
        old_close=TrainingCapture.close
        def close(capture,*args,**kwargs):
            if records:
                i=state['step']-1
                np.savez_compressed(ext.out/f'decoder_chunk_{i//50:04d}.npz',control_index=np.arange(i-len(records)+1,i+1),inputs=np.stack([x['input'] for x in records]),outputs=np.stack([x['output'] for x in records]),targets=np.stack([x['target'] for x in records]));records.clear()
            return old_close(capture,*args,**kwargs)
        TrainingCapture.close=close
        write_json(ext.out/'token_controller.json',{'tokens_file':str(token_file),'tokens_sha256':hashlib.sha256(Path(token_file).read_bytes()).hexdigest(),'decoder_file':str(decoder_file),'decoder_sha256':hashlib.sha256(Path(decoder_file).read_bytes()).hexdigest(),'controller':'stored release encoded_tokens + live10frame proprioception -> matched release decoder ONNX -> all29 joints via deployment PD','frequency_Hz':50,'token_rows':len(tokens),'encoder_rerun':False,'IK_override':False,'online_planner':False,'old_ATM_body_output':'bypassed with zeros; all29 targets replaced by release decoder','history_order':'oldest to newest, grouped omega/q-default/dq/previous_decoder_action/gravity','startup_history':'repeat measured initial state for9 missing prehistory samples; last_action zero','body_actuator_parameters':'deployment C++ values; original simulation force limits and geometry retained','hands':'recorded Revo2 active targets separately','physical_state_writes_during_episode':False,'initialization':'one-time restore from source step830 before first observation; see initial_state_validation.json','source_start_step':830})
    brainco_adapter.install_extension=extension_install
