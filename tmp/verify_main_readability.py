"""Compare CLI parsing/output against the pre-refactor snapshot without CV runs."""
import contextlib
from dataclasses import asdict
import importlib.util
import io
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


before = load('main_before', 'tmp/main_before_readability_refactor.py')
after = load('main_after', 'main.py')
params = dict(camera_matrix='K', dist_coeffs='dist', image_size=[3840,2160])
cases = [
    ['--help'],
    ['--camera','0'],
    ['--video','videos/landing_pad_test.mp4','--headless'],
    ['--camera','1','--output','outputs/example','--width','960','--max-frames','2',
     '--playback-speed','.5','--history-size','12','--max-frame-gap','3',
     '--slow-interval','.8','--local-interval','.1','--pose-reset-after','4','--save-every','10'],
    ['--camera','0','--width','99'],
    ['--camera','0','--playback-speed','nan'],
    ['--camera','0','--max-frames','-1'],
    ['--video','missing_readability_test.mp4'],
    ['--camera','0','--video','videos/landing_pad_test.mp4'],
    [],
]


def invoke(module, arguments):
    stdout, stderr = io.StringIO(), io.StringIO()
    with patch.object(sys,'argv',['main.py',*arguments]), \
         patch.object(module,'load_camera_params',return_value=params) as loader, \
         patch.object(module,'run',return_value=0) as runner, \
         contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            code = module.main()
        except SystemExit as error:
            code = error.code
    config = asdict(runner.call_args.args[0]) if runner.called else None
    return code, stdout.getvalue(), stderr.getvalue(), config, loader.call_args


for case in cases:
    assert invoke(before,case) == invoke(after,case), case
print(f'All {len(cases)} CLI comparisons match the original main.py.')
