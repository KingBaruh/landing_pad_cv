from queue import Empty
from time import perf_counter

import cv2
from detection.slow_detector import SlowDetector
from tracking.recovery import detect_for_recovery
from common.messages import DetectionResult
from common.queue_utils import put_reliable


def slow_process_main(request_queue, result_queue, camera_matrix, pose_limit,
                      config, stop, ready):
    """Detect on requested undistorted frames and reply with their source IDs.

    camera_matrix and pose_limit use the supplied working-image coordinates.
    Every completed request gets a reply, including unsuccessful detections;
    None ends the stream. Return request and detector-call counts at shutdown.
    """
    cv2.setNumThreads(config.opencv_threads)
    detector = SlowDetector(camera_matrix=camera_matrix, max_pose_error_px=pose_limit)
    ready.set()
    requests = calls = 0
    try:
        while not stop.is_set():
            try:
                request = request_queue.get(timeout=.1)
            except Empty:
                continue
            if request is None:
                break
            start = perf_counter()
            detected, scope, count = detect_for_recovery(
                request.frame, detector, request.last_corners, allow_global=request.allow_global)
            requests += 1
            calls += count
            # Keep the original ID/time: Fast must replay a delayed detection
            # before using its corners on a newer frame.
            result = DetectionResult(request.frame_id, request.timestamp, detected.valid,
                                     detected.corners, detected.confidence, scope, count,
                                     (perf_counter()-start)*1000)
            if not put_reliable(result_queue, result, stop):
                break
    finally:
        if stop.is_set():
            result_queue.cancel_join_thread()
    return dict(requests=requests, detector_calls=calls)
