from queue import Empty, Full


def put_latest(queue, item):
    """Keep latency low by dropping stale queued data when necessary."""
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
    """Control messages/requests must not be silently dropped with image data."""
    while not stop.is_set():
        try:
            queue.put(item, timeout=.1)
            return True
        except Full:
            continue
    return False
