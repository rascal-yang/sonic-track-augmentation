"""Keep physics-rate contact latching, record only alerts and aggregate counts."""
import json
import os
from contact_watch_full import ContactWatch as FullWatch


class ContactWatch(FullWatch):
    def __init__(self, env, capture, out):
        super().__init__(env, capture, out)
        self.stream.close()
        (out/'table_contacts_physics.jsonl').unlink()
        self.stream = open(os.devnull, 'w')
        self.positive_samples = 0
        capture.contact_watch = self

    def sample(self):
        # Same sensor reads and latch threshold; avoid dense per-substep JSON.
        values = self.forces()
        total = sum(values.values())
        self.substep_count += 1
        self.peak = max(self.peak, total)
        self.positive_samples += total > .05
        if total > self.threshold and self.latched is None and self.capture.control_index >= self.next_alert_step:
            self.latched = dict(physics_index=self.substep_count-1,
                                control_index=self.capture.control_index,
                                time_s=self.substep_count*self.env.physics_dt,
                                total_table_force_N=total,
                                links_N={k: v for k, v in values.items() if v > .001})
            with (self.out/'contact_alerts.jsonl').open('a') as stream:
                stream.write(json.dumps(self.latched)+'\n')

    def close(self):
        self.stream.close()
