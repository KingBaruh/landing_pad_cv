"""Export silent README demos from saved results; this is not a runtime benchmark.

Run from the repository root: python -m benchmarks.export_demo_videos
Requires FFmpeg on PATH, downloaded source videos, and matching calibration.
"""
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import cv2
import numpy as np

from common.drawing import draw_fast_result


DEMOS = [
    ('v3', 'Viewpoint changes'),
    ('v6', 'Tracking loss and recovery'),
    ('v8_landing', 'Approach to the landing pad'),
]


def export_demo(name, title, params, output):
    run = Path('outputs') / f'experiment_{name}'
    data = json.loads((run / 'results.json').read_text(encoding='utf-8'))
    summary = json.loads((run / 'runtime_summary.json').read_text(encoding='utf-8'))
    size = tuple(data['working_size'])
    original_size = tuple(summary['original_size'])
    K = np.asarray(data['camera_matrix'], dtype=float)
    scaled = params['camera_matrix'].copy()
    scaled[0] *= size[0] / original_size[0]
    scaled[1] *= size[1] / original_size[1]
    if (data['coordinate_space'] != 'undistorted_working_resolution'
            or tuple(params['image_size']) != original_size
            or not np.allclose(K, scaled)):
        raise ValueError('Saved results and calibration geometry do not match.')
    maps = cv2.initUndistortRectifyMap(K, params['dist_coeffs'], None, K,
                                     size, cv2.CV_32FC1)
    rows = {row['frame_id']: row for row in data['frames']}
    source = Path('videos') / f'{name}.mp4'
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError(f'Cannot read {source}; download its Git LFS content.')
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise ValueError('Source FPS is missing.')
    target = output / f'{name}_demo.mp4'
    # Raw BGR stdin contains no audio. Explicit -an also prevents an audio track.
    encoder = subprocess.Popen([
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{size[0]}x{size[1]+60}',
        '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'fast',
        '-crf', '23', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(target),
    ], stdin=subprocess.PIPE)
    count = missing = 0
    poster_saved = False
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if (frame.shape[1], frame.shape[0]) != original_size:
                raise ValueError('Source dimensions differ from the recorded run.')
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            frame = cv2.remap(frame, *maps, cv2.INTER_LINEAR)
            row = rows.get(count)
            if row is not None:
                frame = draw_fast_result(frame, SimpleNamespace(**row))
            else:
                # Never place an older frame's corners on a new image.
                missing += 1
                cv2.rectangle(frame, (4, 4), (850, 54), (25, 25, 25), -1)
                cv2.putText(frame, 'NO SAVED RESULT | source frame skipped by runtime',
                            (15, 36), cv2.FONT_HERSHEY_SIMPLEX, .65,
                            (0, 200, 255), 1, cv2.LINE_AA)
            canvas = cv2.copyMakeBorder(frame, 0, 60, 0, 0,
                                        cv2.BORDER_CONSTANT, value=(25, 25, 25))
            for offset, text in enumerate([
                f'{title} | source time {count/fps:.2f} s | frame {count}',
                'Recorded results replay | original source speed | not a live FPS benchmark',
            ]):
                cv2.putText(canvas, text, (16, size[1]+23+25*offset),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1,
                            cv2.LINE_AA)
            encoder.stdin.write(canvas.tobytes())
            if not poster_saved and count >= fps*2 and row and row['pose_valid']:
                if not cv2.imwrite(str(output / f'{name}_poster.jpg'), canvas):
                    raise OSError('Cannot save demo poster.')
                poster_saved = True
            count += 1
    finally:
        cap.release()
        encoder.stdin.close()
        return_code = encoder.wait()
    if return_code or not poster_saved or count <= max(rows):
        raise RuntimeError(f'Incomplete demo export: {name}')
    result = dict(source=str(source), results=str(run / 'results.json'),
                  video=target.name, source_fps=fps, frames=count,
                  frames_without_saved_result=missing, pose_valid_frames=data['pose_valid_frames'],
                  recorded_tracking_losses=data['losses'], audio=False,
                  presentation='Offline replay of saved results at source speed')
    print(json.dumps(result), flush=True)
    return result


def main():
    cv2.setNumThreads(1)
    output = Path('docs/demos')
    output.mkdir(parents=True, exist_ok=True)
    with np.load('calibration/camera_params.npz') as params:
        manifest = [export_demo(name, title, params, output) for name, title in DEMOS]
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
