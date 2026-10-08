"""Rendering parameters must preserve explicit updates and scene-quality settings."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / 'payload/workspace/pipelines/sonic_teleop_augmentation/render_settings.py'
spec = importlib.util.spec_from_file_location('rt2_settings_under_test', HELPER)
settings = importlib.util.module_from_spec(spec)
spec.loader.exec_module(settings)

class RenderSettingsTests(unittest.TestCase):
    def test_updates_and_first_frame_warmup(self):
        for updates in (1, 2, 4):
            self.assertEqual(settings.updates_for_frame(10, updates), updates)
            self.assertEqual(settings.updates_for_frame(0, updates), max(4, 2*updates))
        self.assertEqual(settings.resolve_updates(None, {}), 1)
        self.assertEqual(settings.resolve_updates(None, {'SONIC_RENDER_UPDATES':'2'}), 2)
        self.assertEqual(settings.resolve_updates(1, {'SONIC_RENDER_UPDATES':'4'}), 1)

    def test_offline_warmup_is_independent_of_measured_updates(self):
        for updates in (1, 2, 4):
            self.assertEqual(settings.offline_update_count(True, True, updates), 20)
            self.assertEqual(settings.offline_update_count(False, True, updates), 4)
            self.assertEqual(settings.offline_update_count(False, False, updates), updates)

    def test_invalid_update_configuration_rejected(self):
        for value in (0, 3, -1, 'bad', 1.5, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                settings.resolve_updates(value, {})

    def test_rt2_startup_retains_single_gpu_and_sync(self):
        args = settings.startup_arguments()
        self.assertIn('--/persistent/rtx/modes/rt2/enabled=true', args)
        self.assertIn('--/renderer/multiGpu/enabled=false', args)
        self.assertIn('--/app/renderer/waitIdle=true', args)

    def test_profile_is_native_dlaa_with_transmission(self):
        # These are the actual settings consumed by the adapters, not a mock renderer.
        actual = settings.RT2_SETTINGS
        self.assertEqual(actual['/rtx/rendermode'], 'RealTimePathTracing')
        self.assertEqual(actual['/rtx/post/aa/op'], 4)
        self.assertEqual(actual['/rtx/rtpt/maxBounces'], 4)
        self.assertEqual(actual['/rtx/rtpt/maxSpecularAndTransmissionBounces'], 6)
        self.assertTrue(actual['/rtx/reflections/enabled'])
        self.assertTrue(actual['/rtx/translucency/enabled'])
        self.assertFalse(actual['/app/asyncRendering'])
        self.assertFalse(actual['/omni/replicator/asyncRendering'])
        self.assertFalse(actual['/rtx-transient/dlssg/enabled'])

if __name__ == '__main__': unittest.main()
