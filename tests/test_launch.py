"""Portable preparation must work without the author's original workspace."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='sonic portable ')
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)/'checkout'
        self.repo.mkdir()
        launcher = ROOT/'launch.py'
        if launcher.exists():
            shutil.copy2(launcher, self.repo/'launch.py')
        payload = self.repo/'payload'
        payload.mkdir()
        (payload/'workspace').mkdir()
        (payload/'workspace/example.json').write_text('{"root":"__RUNTIME__/workspace"}')
        (payload/'bytes.bin').write_bytes(b'preserved source bytes')
        records = {}
        for p in payload.rglob('*'):
            if p.is_file():
                records[str(p.relative_to(payload))] = hashlib.sha256(p.read_bytes()).hexdigest()
        (self.repo/'resources.json').write_text(json.dumps({'files':records}))

    def call(self, *args, env=None):
        return subprocess.run([sys.executable, str(self.repo/'launch.py'), *args],
                              capture_output=True, text=True, env=env)

    def test_materialize_in_new_location_preserves_binary_and_rebases_text(self):
        runtime = Path(self.temp.name)/'new runtime'
        result = self.call('prepare', '--runtime', str(runtime))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((runtime/'bytes.bin').read_bytes(), b'preserved source bytes')
        self.assertEqual(json.loads((runtime/'workspace/example.json').read_text())['root'],
                         str(runtime/'workspace'))

    def test_corrupt_resource_is_rejected_before_runtime_creation(self):
        (self.repo/'payload/bytes.bin').write_bytes(b'corrupt')
        runtime = Path(self.temp.name)/'new runtime'
        result = self.call('prepare', '--runtime', str(runtime))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('checksum', result.stderr.lower())
        self.assertFalse(runtime.exists())

    def test_existing_unrelated_directory_is_not_modified(self):
        runtime = Path(self.temp.name)/'existing'
        runtime.mkdir()
        (runtime/'keep.txt').write_text('keep')
        result = self.call('prepare', '--runtime', str(runtime))
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(list(runtime.iterdir()), [runtime/'keep.txt'])

    def test_upstream_path_substitution_is_rejected(self):
        runtime = Path(self.temp.name)/'robot_runtime'
        result = self.call('prepare', '--runtime', str(runtime))
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(runtime.exists())

    def test_doctor_reports_missing_video_tools_before_collection(self):
        env = dict(os.environ, PATH='')
        runtime = Path(self.temp.name)/'doctor runtime'
        result = self.call('doctor', '--runtime', str(runtime), env=env)
        self.assertNotEqual(result.returncode, 0)
        report = json.loads(result.stdout)
        self.assertIn('ffprobe', report.get('missing_commands', []))
        self.assertIn('ffmpeg', report.get('missing_commands', []))

    def test_runtime_mutation_is_rejected_on_reuse(self):
        runtime = Path(self.temp.name)/'new runtime'
        self.assertEqual(self.call('prepare', '--runtime', str(runtime)).returncode, 0)
        (runtime/'bytes.bin').write_bytes(b'changed')
        result = self.call('prepare', '--runtime', str(runtime))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('checksum', result.stderr.lower())

    def module(self):
        spec = importlib.util.spec_from_file_location('launcher_fixture', self.repo/'launch.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_default_seed_is_independent_of_inherited_environment(self):
        module = self.module()
        previous = os.environ.get('G1_GRASP_SOURCE')
        try:
            os.environ['G1_GRASP_SOURCE'] = '/tmp/stale_seed'
            runtime = Path(self.temp.name)/'new runtime'
            env = module.environment(runtime)
            self.assertEqual(env['G1_GRASP_SOURCE'], str(runtime/
                'workspace/experiments/20260929_sonic_grasp_smooth/candidates/safe025'))
        finally:
            if previous is None:
                os.environ.pop('G1_GRASP_SOURCE', None)
            else:
                os.environ['G1_GRASP_SOURCE'] = previous

    def test_batch_resume_rejects_changed_candidate_manifest(self):
        module = self.module()
        candidates = Path(self.temp.name)/'candidates'
        (candidates/'pose_01').mkdir(parents=True)
        manifest = candidates/'pose_01/prepared_files.json'
        manifest.write_text('{"initial_state.npz":"first"}')
        output = Path(self.temp.name)/'collection'
        bind = getattr(module, 'bind_collection', None)
        self.assertTrue(callable(bind), 'Collection provenance guard is missing')
        bind(candidates, output)
        manifest.write_text('{"initial_state.npz":"changed"}')
        with self.assertRaisesRegex(RuntimeError, 'candidate'):
            bind(candidates, output)


if __name__ == '__main__':
    unittest.main()
