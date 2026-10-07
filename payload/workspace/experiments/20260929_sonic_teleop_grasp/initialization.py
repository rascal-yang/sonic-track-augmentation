"""A single physical-state restore before initial inspection, never during rollout."""
from pathlib import Path
import json
import numpy as np

def install_restore(extension, path):
    import torch
    env = extension.env
    z = np.load(path)
    done = False
    def tensor(x):
        return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=env.device)[None]
    def restore():
        nonlocal done
        assert not done, 'Grasp initialization must run exactly once'
        robot = env.scene['robot']
        order = [z['joint_names'].tolist().index(n) for n in robot.joint_names]
        robot.reset()
        robot.write_root_state_to_sim(tensor(np.r_[z['root_position'], z['root_quaternion_wxyz'], z['root_linear_velocity_world'], z['root_angular_velocity_world']]))
        robot.write_joint_state_to_sim(tensor(z['joint_position'][order]), tensor(z['joint_velocity'][order]))
        robot.set_joint_position_target(tensor(z['joint_position'][order]))
        obj = env.scene['object']
        obj.reset()
        obj.write_root_state_to_sim(tensor(np.r_[z['object_position'], z['object_quaternion_wxyz'], z['object_linear_velocity'], z['object_angular_velocity']]))
        for side in ['left', 'right']:
            prefix = 'brainco_' + side
            hand = env.scene[prefix]
            hi = [z[prefix+'_joint_names'].tolist().index(n) for n in hand.joint_names]
            hand.reset()
            hand.write_root_state_to_sim(tensor(np.concatenate([z[prefix+'_'+key] for key in ['root_position', 'root_quaternion', 'root_linear_velocity', 'root_angular_velocity']])))
            hand.write_joint_state_to_sim(tensor(z[prefix+'_joint_position'][hi]), tensor(z[prefix+'_joint_velocity'][hi]))
            hand.set_joint_position_target(tensor(z[prefix+'_position_target'][hi]))
        env.sim.forward()
        env.scene.update(0.)
        bi = [z['body_names'].tolist().index(n) for n in robot.body_names]
        report = dict(source_step=830, one_time_restore=True,
                      root_max_error_m=float(np.max(abs(robot.data.root_pos_w[0].cpu().numpy()-z['root_position']))),
                      joint_max_error_rad=float(np.max(abs(robot.data.joint_pos[0].cpu().numpy()-z['joint_position'][order]))),
                      body_position_max_error_m=float(np.max(abs(robot.data.body_pos_w[0].cpu().numpy()-z['body_position_world'][bi]))),
                      note='No physics elapsed during restore. Solver warm-start state unavailable; hand root velocities finite-differenced.')
        (extension.out/'initial_state_validation.json').write_text(json.dumps(report, indent=2)+'\n')
        assert report['root_max_error_m'] < 1e-5 and report['joint_max_error_rad'] < 1e-5
        assert report['body_position_max_error_m'] < .001
        done = True
    extension.align_brainco = restore
