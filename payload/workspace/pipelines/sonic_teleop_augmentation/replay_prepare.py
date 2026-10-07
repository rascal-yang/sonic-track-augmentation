"""Freeze executed actions, not measured-motion re-encodings, for a fresh simulator."""
import argparse,shutil
from common import *

def prepare(first,output):
    first=Path(first).resolve();c=Path(output).resolve()
    assert read(first/'collection_result.json')['task_success']
    cfg=read(first/'token_controller.json')
    assert sha(MODEL/'model_decoder.onnx')==cfg['decoder_sha256']
    z=arrays(first/'executed_teleop_tokens.npz')
    d=arrays(first/'decoder_execution.npz')
    assert np.array_equal(z['sonic_encoded_tokens'],d['inputs'][:,:64])
    assert z['sonic_encoded_tokens'].shape==(len(z['time_s']),64)
    z['encoder_joint_names']=np.array(read(first/'token_actuator_setup.json')['joint_names'])
    c.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(c/'grasp_tokens.npz',**{k:v for k,v in z.items() if k not in ['encoder_inputs','decoder_inputs','decoder_outputs','pd_targets']})
    for name in ['initial_state.npz','deployment_parameters.json']:shutil.copy2(first/name,c/name)
    plan=read(first/'plan.json');plan.update(first_execution=str(first),kind='mobile_fixed_token_replay',online_encoder=False,online_planner=False,online_IK=False,
        hypothesis='Stored actual tokens with matched initialization and live decoder history reproduce the first execution; independently test this.',training_ready=False)
    write(c/'plan.json',plan)
    write(c/'provenance.json',dict(first_execution=str(first),decoder_sha256=cfg['decoder_sha256'],
        original_actions_sha256=sha(first/'executed_teleop_tokens.npz'),independent_physics=True))
    write(c/'prepared_files.json',{p.name:sha(p) for p in c.iterdir() if p.is_file()})
    return c

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--first',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();print(prepare(a.first,a.output))
