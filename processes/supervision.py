"""Top-level spawn target: surface failures to the parent and stop peer workers."""
import os
import traceback


def worker_entry(name, target, args, status_queue, stop):
    status_queue.put(dict(worker=name,pid=os.getpid(),event='started'))
    try:
        stats = target(*args)
        status_queue.put(dict(worker=name,pid=os.getpid(),event='finished',stats=stats))
    except KeyboardInterrupt:
        stop.set()
        status_queue.put(dict(worker=name,pid=os.getpid(),event='cancelled'))
    except BaseException:
        stop.set()
        status_queue.put(dict(worker=name,pid=os.getpid(),event='error',error=traceback.format_exc()))
        raise
