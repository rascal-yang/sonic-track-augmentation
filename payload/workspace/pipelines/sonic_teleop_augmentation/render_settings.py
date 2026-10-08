"""RT2 native DLAA profile shared verbatim by both repository adapters.

Updates are render calls at an unchanged physical state, not samples per pixel.
This module does not import Isaac Sim until install_launcher() is requested.
"""
import os

RT2_SETTINGS = {
    '/rtx/rendermode': 'RealTimePathTracing',
    '/persistent/rtx/modes/rt2/enabled': True,
    '/rtx/post/aa/op': 4,
    '/rtx/rtpt/maxBounces': 4,
    '/rtx/rtpt/maxSpecularAndTransmissionBounces': 6,
    '/rtx/rtpt/translucency/virtualMotion/enabled': True,
    '/rtx/reflections/enabled': True,
    '/rtx/translucency/enabled': True,
    '/rtx-transient/dlssg/enabled': False,
    '/omni/replicator/asyncRendering': False,
    '/app/asyncRendering': False,
    '/app/renderer/waitIdle': True,
    '/rtx/post/tonemap/exposureKey': .6,
}


def resolve_updates(value=None, environ=None):
    env = os.environ if environ is None else environ
    value = env.get('SONIC_RENDER_UPDATES', '1') if value is None else value
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError('Render updates must be 1, 2 or 4')
    try:
        result = int(value)
    except ValueError as error:
        raise ValueError('Render updates must be 1, 2 or 4') from error
    if result not in (1, 2, 4):
        raise ValueError('Render updates must be 1, 2 or 4')
    return result


def updates_for_frame(frame_index, updates):
    updates = resolve_updates(updates)
    return max(4, 2*updates) if frame_index == 0 else updates


def offline_update_count(initial, warmup, updates):
    """Keep the historical offline warmup budget separate from measured frames."""
    return 20 if initial else (4 if warmup else resolve_updates(updates))


def startup_arguments():
    return ['--/persistent/rtx/modes/rt2/enabled=true',
            '--/app/renderer/waitIdle=true', '--/renderer/activeGpu=0',
            '--/renderer/multiGpu/enabled=false',
            '--/renderer/multiGpu/autoEnable=false',
            '--/renderer/multiGpu/maxGpuCount=1']


def configure_rt2(settings):
    for key, value in RT2_SETTINGS.items():
        if isinstance(value, bool):
            settings.set_bool(key, value)
        elif isinstance(value, int):
            settings.set_int(key, value)
        elif isinstance(value, float):
            settings.set_float(key, value)
        else:
            settings.set_string(key, value)
    keys = [*RT2_SETTINGS, '/renderer/activeGpu', '/renderer/multiGpu/enabled']
    actual = {key: settings.get(key) for key in keys}
    for key, value in RT2_SETTINGS.items():
        if actual[key] != value:
            raise RuntimeError('Renderer setting did not take effect: '+key)
    return actual


def install_launcher():
    """Select RT2 before Kit starts; camera-free replay keeps its original app."""
    from isaaclab.app import AppLauncher
    original = AppLauncher.__init__
    def initialize(self, args, **kwargs):
        if args.enable_cameras:
            args.renderer = 'RealTimePathTracing'
            args.kit_args += ' ' + ' '.join(startup_arguments())
        original(self, args, **kwargs)
    AppLauncher.__init__ = initialize
