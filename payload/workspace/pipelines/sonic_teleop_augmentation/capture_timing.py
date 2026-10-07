"""Wall-clock instrumentation only; no stepping, synchronization, or data changes."""
import atexit
import functools
import json
import time
from pathlib import Path


def install(capture_class):
    if getattr(capture_class, '_wall_timing_installed', False):
        return
    capture_class._wall_timing_installed = True
    started = time.perf_counter()
    counters = {}
    stack = []
    output = None

    def save():
        if output is None:
            return
        report = dict(elapsed_since_install_s=time.perf_counter()-started,
                      methods=counters,
                      semantics='Inclusive and exclusive wall time; nested capture methods '
                      'are excluded from exclusive totals. GPU work is not explicitly '
                      'synchronized. Rendering includes camera readback and video writing. '
                      'This does not isolate GPU kernel execution time.')
        target = output/'capture_timing.json'
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2)+'\n')
        temporary.replace(target)

    def wrap(name):
        original = getattr(capture_class, name)

        @functools.wraps(original)
        def measured(self, *args, **kwargs):
            nonlocal output
            frame = [time.perf_counter(), 0.0]
            stack.append(frame)
            try:
                return original(self, *args, **kwargs)
            finally:
                elapsed = time.perf_counter()-frame[0]
                stack.pop()
                if stack:
                    stack[-1][1] += elapsed
                counter = counters.setdefault(name, dict(calls=0, inclusive_s=0.0,
                                                         exclusive_s=0.0, max_s=0.0))
                counter['calls'] += 1
                counter['inclusive_s'] += elapsed
                counter['exclusive_s'] += elapsed-frame[1]
                counter['max_s'] = max(counter['max_s'], elapsed)
                if hasattr(self, 'out'):
                    output = Path(self.out)
                if name == 'close' or (name == 'end' and self.control_index % 250 == 0):
                    save()

        setattr(capture_class, name, measured)

    for name in ['__init__', 'begin', 'snapshot', 'render_observation',
                 'post_step', 'end', 'flush_chunks', 'close']:
        wrap(name)
    atexit.register(save)
