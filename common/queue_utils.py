from queue import Empty, Full


def put_latest(queue, item):
    """Try to enqueue current image data without blocking; return success.

    Drop one queued item if full. Multiprocessing feeder/consumer races can
    still make the retry fail, so this is best effort, not guaranteed delivery.
    Use put_reliable for requests, replies and end-of-stream sentinels.
    """
    try:
        queue.put_nowait(item)
        return True
    except Full:
        pass

    try:
        queue.get_nowait()
    except Empty:
        pass

    try:
        queue.put_nowait(item)
        return True
    except Full:
        return False


def put_reliable(queue, item, stop):
    """Retry delivery until queued (True) or cancellation is observed (False).

    Short timeouts let the caller notice stop while another worker is stalled;
    control messages must not use the image queue's deliberate drop policy.
    """
    while not stop.is_set():
        try:
            queue.put(item, timeout=.1)
            return True
        except Full:
            continue
    return False
