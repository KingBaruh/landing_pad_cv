"""Measure Fast on identical cached frames, excluding video decoding and Slow.

Every frame is processed. Slow replies are delivered after a fixed six-frame
delay, so catch-up cost is included without nondeterministic scheduling. This
is compute capacity, not end-to-end FPS; also run main.py on the source video.
"""
import argparse
import json
from pathlib import Path
import sys
from time import monotonic, perf_counter

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibration.calibrate import load_camera_params
from common.messages import FrameMessage, DetectionResult
from config.config import RuntimeConfig
from detection.slow_detector import SlowDetector
from main import _prepare_camera_parameters
from processes.fast_process import record
from processes.runtime_engine import FastEngine
from tracking.recovery import detect_for_recovery


def benchmark(args):
    config = RuntimeConfig(str(args.video), str(args.output), opencv_threads=args.threads)
    cv2.setNumThreads(args.threads)
    original, size, K, dist, limit = _prepare_camera_parameters(
        config, load_camera_params(str(args.camera_params)))
    cache = args.cache
    if not cache.exists():
        cap = cv2.VideoCapture(str(args.video))
        frames = []
        try:
            for _ in range(args.frames):
                ok, frame = cap.read()
                if not ok:
                    break
                if frame.shape[1::-1] != original:
                    raise ValueError('Video size does not match calibration.')
                frames.append(cv2.resize(frame, size, interpolation=cv2.INTER_AREA))
        finally:
            cap.release()
        if not frames:
            raise ValueError('No decoded frames.')
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, np.stack(frames))
    frames = np.load(cache)[:args.frames]
    if frames.shape[2:0:-1] != size:
        raise ValueError('Cached frame size does not match calibration.')
    engine = FastEngine(K, limit, config)
    detector = SlowDetector(camera_matrix=K, max_pose_error_px=limit)
    maps = cv2.initUndistortRectifyMap(K, dist, None, K, size, cv2.CV_32FC1)
    pending = None
    rows = []
    for index, frame in enumerate(frames):
        start = perf_counter()
        msg = FrameMessage(index, index/30, cv2.remap(frame, *maps, cv2.INTER_LINEAR), monotonic())
        engine.advance(msg)
        if pending is not None and index >= pending.frame_id+6:
            engine.accept(pending)
            pending = None
        result = engine.result()
        request = engine.request()
        result.processing_ms = (perf_counter()-start)*1000
        rows.append(record(result))
        if request is not None:
            engine.sent(request)
            detected, scope, calls = detect_for_recovery(
                request.frame, detector, request.last_corners, allow_global=request.allow_global)
            pending = DetectionResult(index, msg.timestamp, detected.valid, detected.corners,
                                      detected.confidence, scope, calls)
    times = np.array([row['processing_ms'] for row in rows])
    summary = dict(frames=len(rows), threads=args.threads, mean_ms=float(times.mean()),
                   p50_ms=float(np.median(times)), p95_ms=float(np.percentile(times, 95)),
                   max_ms=float(times.max()), compute_fps=float(1000/times.mean()),
                   over_33ms=int((times > 1000/30).sum()),
                   valid_poses=sum(row['pose_valid'] for row in rows), **engine.stats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(summary=summary, frames=rows,
                                        events=engine.detection_events), indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True,
                        help='Use a different .npy cache for every video/size.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=180)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--camera-params', type=Path, default=Path('calibration/camera_params.npz'))
    benchmark(parser.parse_args())
