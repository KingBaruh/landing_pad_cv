from dataclasses import fields
import json
from pathlib import Path
from queue import Empty
from time import monotonic, perf_counter

import cv2
import numpy as np
from common.messages import FrameMessage
from common.queue_utils import put_latest, put_reliable
from metrics.logger import MetricsLogger
from processes.runtime_engine import FastEngine


def record(result):
    return {field.name: (value.tolist() if isinstance(value, np.ndarray) else value)
            for field in fields(result) if field.name != 'frame'
            for value in [getattr(result, field.name)]}


def fast_process_main(frame_queue, request_queue, detection_queue, result_queue,
                      K, dist, size, pose_limit, config, stop, ready):
    cv2.setNumThreads(config.opencv_threads)
    maps = cv2.initUndistortRectifyMap(K, dist, None, K, size, cv2.CV_32FC1)
    engine = FastEngine(K, pose_limit, config)
    rows = {}
    output = Path(config.output)
    ready.set()
    began = monotonic()
    eof = False
    try:
        while not stop.is_set():
            try:
                raw = frame_queue.get(timeout=.1)
            except Empty:
                continue
            if raw is None:
                eof = True
                break
            start = perf_counter()
            if (raw.frame.shape[1],raw.frame.shape[0]) != size:
                raise ValueError('Unexpected working frame size.')
            msg = FrameMessage(raw.frame_id, raw.timestamp,
                               cv2.remap(raw.frame, *maps, cv2.INTER_LINEAR), raw.captured_at)
            engine.advance(msg)
            try:
                detection = detection_queue.get_nowait()
            except Empty:
                detection = None
            if detection is not None:
                engine.accept(detection)
            result = engine.result()
            request = engine.request()
            if request is not None and put_reliable(request_queue, request, stop):
                engine.sent(request)
            result.processing_ms = (perf_counter()-start)*1000
            result.latency_ms = (monotonic()-msg.captured_at)*1000
            rows[msg.frame_id] = record(result)
            put_latest(result_queue, result)
        # The last slow response may arrive after the last input frame. Replay
        # it only over existing history; do not relabel it as a later frame.
        deadline = monotonic()+10
        while eof and engine.pending_id is not None and not stop.is_set():
            if monotonic() >= deadline:
                raise TimeoutError('Slow detector did not finish its final request.')
            try:
                detection = detection_queue.get(timeout=.1)
            except Empty:
                continue
            start = perf_counter()
            if engine.accept(detection):
                result = engine.result(count_rejection=False)
                result.processing_ms = (perf_counter()-start)*1000
                rows[result.frame_id] = record(result)
                put_latest(result_queue, result)
        put_reliable(request_queue, None, stop)
        report = dict(processed_frames=len(rows), valid_frames=sum(r['valid'] for r in rows.values()),
                      pose_valid_frames=sum(r['pose_valid'] for r in rows.values()),
                      elapsed_s=monotonic()-began, working_size=list(size),
                      camera_matrix=K.tolist(), pose_error_limit_px=pose_limit,
                      coordinate_space='undistorted_working_resolution',
                      **engine.stats, detection_events=engine.detection_events,
                      frames=list(rows.values()))
        (output/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        logger = MetricsLogger(output/'metrics.csv')
        for row in rows.values():
            logger.log(**{**row, 'processing_time_ms':row['processing_ms']})
        put_reliable(result_queue, None, stop)
        return {key:value for key,value in report.items() if key not in ('frames','detection_events')}
    finally:
        if stop.is_set():
            request_queue.cancel_join_thread()
            result_queue.cancel_join_thread()
