"""Package successful named references, tokens and executed labels with provenance."""
from pathlib import Path
import json,shutil,hashlib,zipfile
root=Path(__file__).resolve().parent;workspace=root.parents[1]
out=workspace/'exports/g1_revo2_25mm_smooth_20260929';out.mkdir(exist_ok=False)
for candidate,run in [('safe025','safe025_001'),('smooth_final025','smooth_final025_001')]:
 r=root/'runs'/run;c=root/'candidates'/candidate
 q=json.loads((r/'quality.json').read_text());assert q['task_success']
 dest=out/candidate;dest.mkdir()
 for name in ['grasp_reference.npz','grasp_tokens.npz','initial_state.npz','deployment_parameters.json','plan.json','waypoints.json','geometry_check.json','provenance.json']:
  if (c/name).exists():shutil.copy2(c/name,dest/name)
 for name in ['executed_trajectory_named.npz','executed_trajectory_schema.json','quality.json','token_summary.json','token_controller.json','contact_validation.json','hand_validation.json','initial_state_validation.json','direct_provenance.json','planner_provenance.json','validation.json','visual_realism_review.json']:
  if (r/name).exists():shutil.copy2(r/name,dest/name)
 for original,name in [('head.mp4','first_person.mp4'),('000000.mp4','third_person.mp4')]:shutil.copy2(r/original,dest/name)
for name in ['RESULTS.md','stage_motion_metrics.json','acceptance_plan.json']:
 shutil.copy2(root/name,out/name)
(out/'README.md').write_text('''# G1/Revo2 25mm grasp references and simulation execution

Only two successful versions are retained: safe025 (fast,11.72s) and smooth_final025 (conservative,15.42s). Each includes reference, encoded SONIC tokens, actual executed states/PD targets, initialization, parameters and validation evidence. Other trials were deleted at the user's request. See RESULTS.md for measured tradeoffs.

All are grasp-only from the original source step830, not walk-to-grasp. Control50Hz;64float32 release SONIC latent values plus separate Revo2 commands. `waypoints.json` defines7 right-arm reference postures; it is not the actuator command stream. `executed_trajectory_named.npz` contains actual pre-action states and matched PD targets with explicit joint names.43body columns include14 legacy compatibility joints; select the29physical body joints by name. Hands are separate.

For simulation decoder replay, load grasp_tokens.npz with the matched release decoder and live10frame history, deployment joint mapping/gains and recorded Revo2 targets. Model paths/hashes are in provenance.json. Models, scene assets and runtime are not included in this data archive. Initialization is a simulation snapshot, not an instruction to set a real robot state. Full experiment and replay code: experiments/20260929_sonic_grasp_smooth.

Videos are unretimed Isaac RTX realtime/DLAA previews, notRT2. Native-rate temporal quality is not certified; training_ready=false. No physical perturbation study or real-hardware replay validation is claimed. RESULTS.md contains workspace-relative evidence links; corresponding packaged metrics/videos are in each version folder.
''')
source=out/'experiment_scripts';source.mkdir()
for script in root.glob('*.py'):shutil.copy2(script,source/script.name)
manifest={str(f.relative_to(out)):hashlib.sha256(f.read_bytes()).hexdigest() for f in out.rglob('*') if f.is_file()}
(out/'SHA256SUMS.json').write_text(json.dumps(manifest,indent=2)+'\n')
archive=out.with_suffix('.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for f in out.rglob('*'):
  if f.is_file():z.write(f,f.relative_to(out.parent))
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
print(archive,archive.stat().st_size)
