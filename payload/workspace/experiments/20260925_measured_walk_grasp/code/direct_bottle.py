"""Natural standing plus fresh GPT waypoints; no trajectory hints or ONNX planning."""
import numpy as np
from stationary_hint import StationaryHintPlanner
from experiment import write_json

class DirectBottlePlanner(StationaryHintPlanner):
    def __init__(self,env,out,sonic):
        super().__init__(env,out,sonic)
        self.phase='grasp';self.grail_gain=0.
        write_json(out/'direct_provenance.json',{'historical_trajectory_loaded':False,'historical_waypoint_file_loaded':False,'SONIC_ONNX_calls':0,'GRAIL_applied_residual':0,'initial_pose':'MJCF standing joints with symmetric relaxed arms; no motion replay','control':'Current GPT/Codex decisions -> actual-state right-arm IK, bounded gravity compensation and command speed; SONIC other joints -> physical G1/Revo2','external_GPT6_API':False})
    def observation(self):
        s=super().observation();r=self.env.scene['robot'];p=r.data.root_pos_w[0].cpu().numpy()
        s.update(phase=self.phase,grail_phase_gain=0.,planner_calls=0,navigation_error_m=float(np.linalg.norm(p[:2]-self.reference_origin[:2])),root_linear_velocity_m_s=r.data.root_lin_vel_w[0].cpu().tolist(),feet={})
        for side in ['left','right']:
            i=r.body_names.index(side+'_ankle_roll_link');s['feet'][side]={'position_m':r.data.body_pos_w[0,i].cpu().tolist(),'linear_velocity_m_s':r.data.body_lin_vel_w[0,i].cpu().tolist()}
        return s
