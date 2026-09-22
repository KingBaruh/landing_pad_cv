"""Profile real worker processes using the same CLI as main.py.

Example: python benchmarks/profile_runtime.py --video videos/v3.mp4 --headless
         --max-frames 180 --output outputs/fps_review/profile_v3

Profiling adds overhead. Use normal main.py runs for final throughput numbers.
"""
import cProfile
from pathlib import Path
import pstats
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main
from processes.fast_process import fast_process_main
from processes.video_player import video_player_main


def _profile(target, args, config_index, name):
    profiler = cProfile.Profile()
    try:
        return profiler.runcall(target, *args)
    finally:
        output = Path(args[config_index].output)
        output.mkdir(parents=True, exist_ok=True)
        profiler.dump_stats(str(output / f'{name}.prof'))
        with (output / f'{name}_profile.txt').open('w', encoding='utf-8') as stream:
            pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats('cumulative').print_stats(50)


def profile_fast(*args):
    return _profile(fast_process_main, args, 8, 'fast')


def profile_video(*args):
    return _profile(video_player_main, args, 4, 'video')


if __name__ == '__main__':
    main.fast_process_main = profile_fast
    main.video_player_main = profile_video
    raise SystemExit(main.main())
