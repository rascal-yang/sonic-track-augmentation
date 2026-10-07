"""200 Hz table-contact audit; latch an early GPT checkpoint within 20 ms."""
import json
import numpy as np


class ContactWatch:
    def __init__(self, env, capture, out):
        self.env, self.capture, self.out = env, capture, out
        self.threshold = .5
        self.latched = None
        self.next_alert_step = 0
        self.substep_count = 0
        self.peak = 0.
        self.stream = (out / 'table_contacts_physics.jsonl').open('w')
        self.sensors = []
        from grasp_extensions import FINGER_LINKS
        for i, link in enumerate(FINGER_LINKS):
            self.sensors.append((link, env.scene.sensors[f'grasp_finger_{i}'], 1))
        for name, sensor in env.scene.sensors.items():
            if name.startswith(('table_watch_','obstacle_watch_')):
                self.sensors.append(((('room:' if name.startswith('obstacle_watch_') else 'table:')+sensor.cfg.prim_path.rsplit('/',1)[-1]), sensor, 0))
        old_update = env.scene.update

        def update(*args, **kwargs):
            result = old_update(*args, **kwargs)
            if self.capture.active:
                self.sample()
            return result
        env.scene.update = update
        (out / 'contact_watch_config.json').write_text(json.dumps({
            'frequency_hz': 1 / env.physics_dt, 'threshold_total_table_force_N': self.threshold,
            'sensor_count': len(self.sensors), 'links': [x[0] for x in self.sensors],
            'response': 'Latch on physics update; pause at next control checkpoint, at most 20 ms later. No state teleport or actuator overwrite.',
            'after_acknowledgement': 'All contacts remain logged; next alert allowed after 0.2s for recovery motion.'}, indent=2))

    def forces(self):
        import torch
        # Preserve every 200 Hz sample, synchronize GPU once for the sensor batch.
        values=torch.stack([sensor.data.force_matrix_w[0,0,channel] for _,sensor,channel in self.sensors]).detach().cpu().numpy()
        return {name:float(np.linalg.norm(v)) for (name,_,_),v in zip(self.sensors,values)}

    def sample(self):
        values = self.forces()
        total = sum(values.values())
        row = {'physics_index': self.substep_count,
               'control_index': self.capture.control_index,
               'time_s': (self.substep_count + 1) * self.env.physics_dt,
               'total_table_force_N': total,
               'links_N': {k: v for k, v in values.items() if v > .001}}
        self.stream.write(json.dumps(row) + '\n')
        self.substep_count += 1
        self.peak = max(self.peak, total)
        if total > self.threshold and self.latched is None and self.capture.control_index >= self.next_alert_step:
            self.latched = row
            with (self.out / 'contact_alerts.jsonl').open('a') as f:
                f.write(json.dumps(row) + '\n')
            self.stream.flush()

    def observation(self):
        return {'alert': self.latched, 'current_table_forces_N': self.forces(),
                'peak_total_table_force_N': self.peak, 'physics_samples': self.substep_count}

    def acknowledge(self, step):
        if self.latched is not None:
            self.next_alert_step = step + 10
        self.latched = None

    def close(self):
        self.stream.close()
