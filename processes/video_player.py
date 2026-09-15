from pathlib import Path
from queue import Empty
from time import monotonic

import cv2
from common.messages import FrameMessage
from common.queue_utils import put_latest, put_reliable
from common.drawing import draw_fast_result


def video_player_main(frame_queue, result_queue, original_size, working_size,
                      config, stop, fast_ready, slow_ready):
    """Read a video/camera, send resized frames to Fast and display its results.

    Sizes are (width, height). Raw input must match the calibration dimensions.
    Return decoding/display counters; raise on capture or preview-save errors.
    """
    cv2.setNumThreads(config.opencv_threads)
    while not (fast_ready.is_set() and slow_ready.is_set()):
        if stop.wait(.05):
            return dict(decoded_frames=0)
    cap = cv2.VideoCapture(config.source)
    if not cap.isOpened():
        raise ValueError(f'Cannot open video: {config.source}')
    if isinstance(config.source,int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,original_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,original_size[1])
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.
    start = monotonic()
    index = rejected_puts = displayed = 0
    last_preview = -config.save_every
    last_result = None
    ended = False
    output = Path(config.output)
    def consume():
        nonlocal ended, last_result, displayed, last_preview
        while True:
            try:
                result = result_queue.get_nowait()
            except Empty:
                break
            if result is None:
                ended = True
                break
            last_result = result
            displayed += 1
            # Draw on the result's own undistorted frame, never a newer raw one.
            preview = draw_fast_result(result.frame.copy(),result)
            if result.frame_id-last_preview >= config.save_every:
                if not cv2.imwrite(str(output/f'frame_{result.frame_id:06d}.jpg'),preview):
                    raise OSError('Could not save preview.')
                last_preview = result.frame_id
            if not config.headless:
                cv2.imshow('Landing Pad Tracking',preview)
    def ui():
        if not config.headless:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop.set()
    try:
        while not stop.is_set() and (not config.max_frames or index < config.max_frames):
            ok, frame = cap.read()
            if not ok:
                break
            if (frame.shape[1],frame.shape[0]) != original_size:
                raise ValueError(f'Calibration size {original_size} != video {(frame.shape[1],frame.shape[0])}')
            due = start+index/fps/config.playback_speed
            while monotonic() < due and not stop.is_set():
                consume()
                ui()
                stop.wait(min(.005,max(0,due-monotonic())))
            if stop.is_set():
                break
            # Latency begins after decoding and playback pacing, before resize.
            # It therefore excludes decoder time and is not display latency.
            captured_at = monotonic()
            if working_size != original_size:
                frame = cv2.resize(frame,working_size,interpolation=cv2.INTER_AREA)
            # Source time stays index/fps even when playback is slower or faster;
            # monotonic wall time is a separate clock used for measurements.
            msg = FrameMessage(index,index/fps,frame,captured_at)
            rejected_puts += int(not put_latest(frame_queue,msg))
            consume()
            if last_result is None and not config.headless:
                preview = frame.copy()
                cv2.putText(preview,'SEARCHING | waiting for first result',(12,28),
                            cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,255),2)
                cv2.imshow('Landing Pad Tracking',preview)
            ui()
            index += 1
        if index == 0 and not stop.is_set():
            raise ValueError('Video contains no readable frames.')
        # EOF must reach Fast even if its input queue is full. Keep consuming
        # results until Fast acknowledges completion with its own sentinel.
        put_reliable(frame_queue,None,stop)
        while not ended and not stop.is_set():
            consume()
            ui()
            stop.wait(.01)
    finally:
        cap.release()
        if not config.headless:
            cv2.destroyAllWindows()
        if stop.is_set():
            frame_queue.cancel_join_thread()
    return dict(decoded_frames=index, source_fps=fps, failed_frame_puts=rejected_puts,
                displayed_results=displayed, elapsed_s=monotonic()-start)
