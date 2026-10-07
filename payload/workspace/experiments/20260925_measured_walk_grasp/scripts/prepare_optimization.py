"""Versioned arm-only acceleration bound; never filter SONIC balance outputs."""
from pathlib import Path
B=Path(__file__).resolve().parents[1]
p=B/'code/grasp_extensions.py';s=p.read_text()
s=s.replace("solved[arm_ids]=previous[arm_ids]+np.clip(solved[arm_ids]-previous[arm_ids],-.02,.02)","""desired_delta=np.clip(solved[arm_ids]-previous[arm_ids],-.02,.02)
            optimized=os.environ.get('G1_OPTIMIZED')=='1'
            if optimized:
                # Bound velocity and acceleration of only the separately controlled arm.
                # Task-space quintic progress still determines approach/contact speed.
                max_velocity=float(getattr(self,'arm_speed_limit',.8))
                desired_delta=np.clip(desired_delta,-max_velocity*.02,max_velocity*.02)
                last_delta=getattr(self,'last_arm_delta',np.zeros(len(arm_ids)))
                max_delta_change=3.0*.02**2
                desired_delta=last_delta+np.clip(desired_delta-last_delta,-max_delta_change,max_delta_change)
                self.last_arm_delta=desired_delta.copy()
            solved[arm_ids]=previous[arm_ids]+desired_delta""")
s=s.replace("if p.get('controller'):","if 'arm_speed_limit_rad_s' in p:\n            self.arm_speed_limit=float(np.clip(p['arm_speed_limit_rad_s'],.2,1.))\n        if p.get('controller'):")
p.write_text(s)
p=B/'scripts/collect.py';s=p.read_text().replace("from experiment import run","os.environ['G1_OPTIMIZED']='0' if a.name in ['stationary_001','stationary_002'] else '1'\nfrom experiment import run");p.write_text(s)
print('Prepared next-run optimization; running baseline is unchanged.')
