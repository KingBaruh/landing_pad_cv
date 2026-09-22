"""Bounded single-producer image queue with shared pixel buffers.

Only small metadata crosses a multiprocessing pipe. Slot ownership lasts until
the reader has copied the image, so an old frame can never acquire newer pixels.
The producer may discard queued messages through get_nowait (put_latest).
"""
from dataclasses import dataclass, replace
from queue import Full

import numpy as np


@dataclass
class _SharedFrame:
    metadata: object
    slot: int


class SharedFrameQueue:
    def __init__(self, context, size, maxsize=1):
        width, height = size
        self.shape = (height, width, 3)
        # Pending messages plus one reader snapshot and one producer write.
        self.slots = maxsize+2
        self.pixels = context.RawArray('B', self.slots*height*width*3)
        self.in_use = context.RawArray('b', self.slots)
        self.slot_lock = context.Lock()
        self.capacity = context.BoundedSemaphore(maxsize)
        self.metadata_queue = context.Queue()

    def _images(self):
        # Construct views in each worker, never pickle a NumPy view of this RAM.
        return np.frombuffer(self.pixels, np.uint8).reshape((self.slots, *self.shape))

    def _release_slot(self, slot):
        with self.slot_lock:
            self.in_use[slot] = 0

    def put(self, item, block=True, timeout=None):
        if not self.capacity.acquire(block, timeout):
            raise Full
        slot = None
        try:
            if item is not None and getattr(item, 'frame', None) is not None:
                frame = item.frame
                if frame.shape != self.shape or frame.dtype != np.uint8:
                    raise ValueError('Shared image queue requires working-size uint8 BGR frames.')
                with self.slot_lock:
                    slot = next((i for i in range(self.slots) if not self.in_use[i]), None)
                    if slot is None:
                        raise Full
                    self.in_use[slot] = 1
                np.copyto(self._images()[slot], frame)
                item = _SharedFrame(replace(item, frame=None), slot)
            self.metadata_queue.put_nowait(item)
        except BaseException:
            if slot is not None:
                self._release_slot(slot)
            self.capacity.release()
            raise

    def put_nowait(self, item):
        self.put(item, block=False)

    def get(self, block=True, timeout=None):
        item = self.metadata_queue.get(block, timeout)
        self.capacity.release()
        if not isinstance(item, _SharedFrame):
            return item
        try:
            # The returned array belongs to the consumer/history, not the ring.
            frame = self._images()[item.slot].copy()
        finally:
            self._release_slot(item.slot)
        return replace(item.metadata, frame=frame)

    def get_nowait(self):
        return self.get(block=False)

    def cancel_join_thread(self):
        self.metadata_queue.cancel_join_thread()

    def close(self):
        self.metadata_queue.close()
