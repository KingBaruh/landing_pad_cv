"""Lossless, geometry-preserving input cache for recorded-video experiments.

Preparation is an offline step, never included in real-time throughput claims.
Frames use exactly the runtime's INTER_AREA resize, with no crop or resampling
in time. Calibration continues to refer to the original capture mode.
"""
import hashlib
import json
from pathlib import Path
from time import perf_counter

import cv2


def prepare_video(source, original_size, working_size, cache_directory):
    source = Path(source).resolve()
    digest = hashlib.sha256()
    with source.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    identity = dict(source_sha256=digest.hexdigest(), original_size=list(original_size),
                    working_size=list(working_size), interpolation='INTER_AREA', codec='HFYU', version=2)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    directory = Path(cache_directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f'{source.stem}_{key}.avi'
    manifest_path = destination.with_suffix('.json')
    if destination.is_file() and manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            manifest = {}  # An interrupted manifest write is not a usable cache.
        if (manifest.get('identity') == identity
                and manifest.get('file_bytes') == destination.stat().st_size):
            print(f'Using prepared video: {destination}', flush=True)
            return destination

    print('Preparing recorded video at working resolution (offline, lossless HuffYUV)...', flush=True)
    began = last_progress = perf_counter()
    capture = cv2.VideoCapture(str(source))
    writer = None
    temporary = destination.with_suffix('.partial.avi')
    count = 0
    try:
        if not capture.isOpened():
            raise ValueError(f'Cannot open video: {source}')
        fps = capture.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            raise ValueError('Video preparation requires a valid source FPS.')
        writer = cv2.VideoWriter(str(temporary), cv2.CAP_FFMPEG,
                                 cv2.VideoWriter_fourcc(*'HFYU'), fps, working_size)
        if not writer.isOpened():
            raise RuntimeError('FFmpeg/HuffYUV support is required for lossless preparation.')
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape[1::-1] != tuple(original_size):
                raise ValueError(f'Calibration size {original_size} != video {frame.shape[1::-1]}')
            if tuple(working_size) != tuple(original_size):
                frame = cv2.resize(frame, working_size, interpolation=cv2.INTER_AREA)
            writer.write(frame)
            count += 1
            if perf_counter()-last_progress >= 5:
                print(f'  Prepared {count} frames...', flush=True)
                last_progress = perf_counter()
        if not count:
            raise ValueError('Video contains no readable frames.')
        # FFmpeg can estimate this from duration and nominal FPS (e.g. a phone
        # variable-rate clip reports 412 but decodes 408). Validate the output
        # against actual successful reads, not that container estimate.
        reported_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    # Publish only after encoding succeeds. An incomplete file has no manifest
    # and cannot be mistaken for a reusable cache on the next run.
    check = cv2.VideoCapture(str(temporary))
    try:
        if not check.isOpened() or int(check.get(cv2.CAP_PROP_FRAME_COUNT)) != count:
            raise OSError('Prepared video frame count did not match the source.')
    finally:
        check.release()
    temporary.replace(destination)
    elapsed = perf_counter()-began
    manifest = dict(identity=identity, source=str(source), frames=count, source_fps=fps,
                    source_reported_frames=reported_count,
                    preparation_s=elapsed, file_bytes=destination.stat().st_size)
    manifest_temporary = manifest_path.with_suffix('.partial.json')
    manifest_temporary.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    manifest_temporary.replace(manifest_path)
    print(f'Prepared {count} frames in {elapsed:.1f}s: {destination}', flush=True)
    return destination
