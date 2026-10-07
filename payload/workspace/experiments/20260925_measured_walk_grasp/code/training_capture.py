"""Synchronous pre-action observations and actual per-substep actuator commands.

No state(t+1) is relabelled as action(t). Video frames correspond to pre-action
states at control indices 0,2,... . Each training frame has TWO 50Hz actions.
"""
import json
from pathlib import Path
import numpy as np
import subprocess, sys
from types import SimpleNamespace
import imageio.v2 as imageio
from scipy.spatial.transform import Rotation


def cpu(value):
    return value.detach().cpu().numpy().copy() if hasattr(value,'detach') else np.asarray(value).copy()


class TrainingCapture:
    def __init__(self, env, out):
        from pxr import UsdGeom, Gf
        import omni.replicator.core as rep
        import omni.usd
        import carb
        self.env=env;self.out=Path(out);self.robot=env.scene['robot'];self.control_index=0;self.substep=0;self.active=False;self.closed=False
        self.spool=self.out/'capture_chunks';self.spool.mkdir();self.chunk_id=0;self.f=SimpleNamespace(attrs={});self.f.attrs['schema']='g1_capture_v1';self.f.attrs['dt']=env.step_dt;self.f.attrs['physics_dt']=env.physics_dt
        self.f.attrs['joint_names']=json.dumps(self.robot.joint_names);self.f.attrs['body_names']=json.dumps(self.robot.body_names)
        self.f.attrs['task']='Pick up the rigid water bottle toy from the window-side desk and hold it above the desk.'
        self.f.attrs['quaternion_order']='wxyz';self.f.attrs['world_axes']='right-handed, Z up, meters';self.f.attrs['camera_axes']='OpenCV x right y down z forward'
        self.f.attrs['action_semantics']='absolute joint_position_target rad sent to actuator model; low-level controls at 50Hz; all physics-substep targets/efforts retained'
        self.f.attrs['torque_note']='applied_torque is actuator-model output; implicit finger solver reaction torques are not direct measured sensor signals'
        self.f.attrs['controller']='SONIC V2 ONNX walking; 3s reference handoff +1s standing; actual-state right-arm IK and bounded gravity compensation, SONIC other joints; GRAIL residual zero throughout. GPT/Codex designs fixed feedback protocol; no external GPT6 API. Session inspects initial/approach/final images; automatic supervisor executes identical guarded strategy across starts.'
        self.f.attrs['action_delay_control_steps']=0
        self.datasets={};self.ann={};self.ops={};self.videos={};self.camera_info={};self.last_rgb={}
        self.stage=omni.usd.get_context().get_stage()
        rep.settings.set_render_rtx_realtime(antialiasing='DLAA')
        carb.settings.get_settings().set_bool('/rtx/translucency/enabled',True)
        carb.settings.get_settings().set_bool('/rtx/reflections/enabled',True)
        carb.settings.get_settings().set_float('/rtx/post/tonemap/exposureKey',.6)
        specs={'head':('torso_link',[.095,0,.45],35), 'right_wrist':('right_wrist_yaw_link',[.035,0,.085],0)}
        self.specs=specs
        for name,(body,offset,pitch) in specs.items():
            path='/World/Capture/'+name;cam=UsdGeom.Camera.Define(self.stage,path)
            cam.CreateHorizontalApertureAttr(36);cam.CreateVerticalApertureAttr(27);cam.CreateFocalLengthAttr(18);cam.CreateClippingRangeAttr((.015,40));self.ops[name]=cam.AddTransformOp()
            product=rep.create.render_product(path,(640,480));annotations={}
            for kind in (['rgb','distance_to_image_plane','semantic_segmentation'] if name=='head' else ['rgb']):
                ann=rep.AnnotatorRegistry.get_annotator(kind,device='cpu');ann.attach([product]);annotations[kind]=ann
            self.ann[name]=annotations
            self.videos[name]=imageio.get_writer(self.out/f'{name}.mp4',fps=25,codec='libx264',quality=8,pixelformat='yuv420p',macro_block_size=1)
            self.camera_info[name]={'body_name':body,'body_local_translation_m':offset,'width':640,'height':480,'K':[[320,0,320],[0,320,240],[0,0,1]],'distortion':[0,0,0,0,0],'depth':'distance to image plane, metres; 0 encodes invalid','simulated_mount':True,'head_pitch_down_deg':pitch,'frame_convention':'OpenCV','world_T_camera':'stored for every frame'}
        self.head_id=self.robot.body_names.index('head_link');self.wrist_id=self.robot.body_names.index('right_hand_palm_link')
        # Wrap the actual writer, AFTER actuator computation, BEFORE PhysX integrates.
        original_write=self.robot.write_data_to_sim
        def write():
            result=original_write()
            if self.active:
                d=self.robot.data
                record={'control_index':self.control_index,'substep_index':self.substep,
                        'timestamp':self.control_index*env.step_dt+self.substep*env.physics_dt,
                        'joint_position_target':cpu(d.joint_pos_target[0]),'joint_velocity_target':cpu(d.joint_vel_target[0]),
                        'joint_effort_command':cpu(d.joint_effort_target[0]),'applied_effort':cpu(self.robot._joint_effort_target_sim[0]),
                        'implicit_position_target':cpu(self.robot._joint_pos_target_sim[0]),
                        'joint_stiffness':cpu(d.joint_stiffness[0]),'joint_damping':cpu(d.joint_damping[0])}
                for key,value in record.items():self.append('physics/'+key,value)
                if self.substep==0:
                    for key in ['joint_position_target','joint_velocity_target','joint_effort_command','joint_stiffness','joint_damping']:self.append('action/'+key,record[key])
                    self.append('action/processed_env_action',cpu(env.action_manager.action[0]))
                self.substep+=1
            return result
        self.robot.write_data_to_sim=write
        (self.out/'camera_calibration.json').write_text(json.dumps(self.camera_info,indent=2))
        self.f.attrs['camera_calibration']=json.dumps(self.camera_info)

    def append(self,key,value):
        self.datasets.setdefault(key,[]).append(np.asarray(value).copy())

    def flush_chunks(self):
        if self.datasets:
            np.savez_compressed(self.spool/f'chunk_{self.chunk_id:05d}.npz', **{k:np.stack(v) for k,v in self.datasets.items()})
            self.chunk_id+=1;self.datasets={}
        (self.spool/'attrs.json').write_text(json.dumps(self.f.attrs))

    def snapshot(self,prefix):
        d=self.robot.data;o=self.env.scene['object'].data
        fields={'timestamp':self.control_index*self.env.step_dt,'control_index':self.control_index,
                'joint_position':cpu(d.joint_pos[0]),'joint_velocity':cpu(d.joint_vel[0]),'applied_torque':cpu(d.applied_torque[0]),
                'root_position':cpu(d.root_pos_w[0]),'root_quaternion_wxyz':cpu(d.root_quat_w[0]),
                'root_linear_velocity_world':cpu(d.root_lin_vel_w[0]),'root_angular_velocity_world':cpu(d.root_ang_vel_w[0]),
                'body_position_world':cpu(d.body_pos_w[0]),'body_quaternion_wxyz':cpu(d.body_quat_w[0]),
                'object_position':cpu(o.root_pos_w[0]),'object_quaternion_wxyz':cpu(o.root_quat_w[0]),
                'object_linear_velocity':cpu(o.root_lin_vel_w[0]),'object_angular_velocity':cpu(o.root_ang_vel_w[0])}
        for k,v in fields.items():self.append(prefix+'/'+k,v)
        if hasattr(self.env,'_grasp_extension'):
            metrics=self.env._grasp_extension.metrics()
            for k,v in metrics.items():
                if isinstance(v,(float,int,bool)):self.append(prefix+'/metrics/'+k,v)
        return fields

    def render_observation(self):
        from pxr import Gf
        d=self.robot.data
        for name,(body,offset,pitch) in self.specs.items():
            i=self.robot.body_names.index(body);q=cpu(d.body_quat_w[0,i]);r=Rotation.from_quat(q[[1,2,3,0]])
            eye=cpu(d.body_pos_w[0,i])+r.apply(offset)
            if name=='head':
                direction=r.apply([np.cos(np.deg2rad(pitch)),0,-np.sin(np.deg2rad(pitch))]);up=r.apply([0,0,1])
            else:
                # Fixed palm-mounted optical axis, not look-at-object privileged targeting.
                direction=r.apply([1.,0,-.30]);up=r.apply([0,1.,0])
            matrix=Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*(eye+direction)),Gf.Vec3d(*up)).GetInverse();self.ops[name].Set(matrix)
            self.append('camera/'+name+'/world_T_camera_cv',np.array(matrix).T@np.diag([1,-1,-1,1]))
        # Flush all dynamic mesh transforms and new camera transforms; no physics step.
        for _ in range(4 if self.control_index==0 else 2):self.env.sim.render()
        if self.control_index==0:self.stage.Export(str(self.out/'isaac_initial_scene.usda'))
        for name,anns in self.ann.items():
            for kind,ann in anns.items():
                output=ann.get_data();info=output.get('info',{}) if isinstance(output,dict) else {}
                a=np.asarray(output['data'] if isinstance(output,dict) else output)
                if kind=='rgb':
                    if a.shape[:2]!=(480,640) or a.ndim!=3:raise RuntimeError(f'Invalid synchronized {name} RGB {a.shape}')
                    a=a[:,:,:3].copy();self.last_rgb[name]=a;self.videos[name].append_data(a);imageio.imwrite(self.out/f'latest_{name}.png',a)
                elif kind=='distance_to_image_plane':
                    a=a.reshape(480,640).astype('float32');valid=np.isfinite(a)&(a>0)&(a<40)
                    self.append('camera/'+name+'/depth_valid',valid);a=np.where(valid,a,0).astype('float32')
                else:
                    a=a.reshape(480,640).astype('uint32')
                    if info:(self.out/'semantic_labels.json').write_text(json.dumps(info,default=lambda v:np.asarray(v).tolist(),indent=2))
                self.append('camera/'+name+'/'+kind,a)
            self.append('camera/'+name+'/timestamp',self.control_index*self.env.step_dt);self.append('camera/'+name+'/control_index',self.control_index)

    def begin(self):
        if 'brainco_right' in self.env.scene.articulations:
            for side in ['left','right']:
                h=self.env.scene['brainco_'+side]
                self.f.attrs['brainco_'+side+'_joint_names']=json.dumps(h.joint_names)
                self.f.attrs['brainco_'+side+'_body_names']=json.dumps(h.body_names)
                for name,value in [('joint_position',h.data.joint_pos[0]),('joint_velocity',h.data.joint_vel[0]),('position_target',h.data.joint_pos_target[0]),('body_position',h.data.body_pos_w[0]),('body_quaternion_wxyz',h.data.body_quat_w[0])]:
                    self.append('brainco/'+side+'/'+name,cpu(value))
        self.snapshot('observation')
        if self.control_index%2==0:self.render_observation()
        self.substep=0;self.active=True

    def post_step(self):
        self.control_index+=1
        self.snapshot("next_observation")
        self.control_index-=1

    def end(self):
        self.active=False
        assert self.substep==self.env.cfg.decimation,(self.substep,self.env.cfg.decimation)
        self.control_index+=1
        if self.control_index%50==0:self.flush_chunks()

    def close(self):
        if self.closed:return
        self.active=False;self.f.attrs['control_steps']=self.control_index
        self.flush_chunks()
        for writer in self.videos.values():writer.close()
        subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'scripts/pack_capture.py'),str(self.out)],check=True)
        self.closed=True
