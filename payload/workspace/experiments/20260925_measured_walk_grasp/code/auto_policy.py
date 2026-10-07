"""Explicit geometric supervisor for reproducible collection; not a GPT API."""
import numpy as np
from scipy.spatial.transform import Rotation

class AutoGraspPolicy:
    def __init__(self):
        self.target=None;self.phase='navigate';self.attempts=0;self.align_rounds=0
        self.euler=[-1.57,0,2.1]
    def decide(self,o):
        step=o['step'];g=o['grasp_information'];m=g['metrics']
        q=np.asarray(o['object_quat_wxyz']);w=Rotation.from_quat(q[[1,2,3,0]]).apply([-.108,.005,.16])+o['object_pos']
        pad=np.asarray(g['pad_based_grasp_center_m']);palm=np.asarray(o['palm_positions']['right'])
        if step==0:
            return {'action':'navigate','steps':200,'goal':{'root_xy':[-.10,-.46],'yaw':np.pi/2,'speed':.20},'reason':'Approach photographed window desk using SONIC planner.'}
        if step==200:
            self.target=w+np.array([.035,-.075,.13]);self.phase='align'
            action='reach';reason='Half-open hand approaches the semantic wrist grasp region.'
        elif self.phase in ['closed','lift','hold'] and m['opposed_digits_contact']:
            if m['whole_object_clearance_m']>.035 and m['object_table_normal_force_N']<.1:
                self.phase='hold';return {'action':'continue','steps':100,'goal':{},'reason':'Whole toy clear of table with opposed finger contact; hold and recheck.'}
            self.target[2]+=.06;self.phase='lift';action='lift';reason='Opposed contacts retained; fixed XY test lift of6cm.'
        else:
            if self.phase in ['closed','lift','hold']:
                self.phase='align';self.target=w-Rotation.from_euler('xyz',self.euler).apply([.066,.075,0]);self.align_rounds=0
                action='reach';reason='Lost enclosure; reopen halfway and approach current toy wrist region.'
            else:
                delta=w-pad
                self.target+=np.clip(delta,-.04,.04);self.align_rounds+=1
                ready=np.linalg.norm(delta)<.025 or m['opposed_digits_contact']
                if ready and self.attempts<3:
                    action='close_hand';self.phase='closed';self.attempts+=1;reason='Current geometry or opposed contact supports enclosure; close fingers.'
                else:action='reach';reason='Correct measured fingertip-center error, capped4cm per axis.'
        closure={'reach':.5,'close_hand':.85,'lift':1}[action]
        goal={'hand':'right','target':self.target.tolist(),'wrist_euler_xyz':self.euler,'duration_steps':95,'closure':closure,'closure_steps':70}
        if action=='close_hand':goal['finger_stiffness']=4
        if action=='lift':goal['lift_m']=.06
        return {'action':action,'steps':100,'goal':goal,'waypoint_sequence':[{'target':self.target.tolist(),'wrist_euler_xyz':self.euler,'duration_steps':95},{'target':self.target.tolist(),'wrist_euler_xyz':self.euler,'duration_steps':200}],'reason':reason}
