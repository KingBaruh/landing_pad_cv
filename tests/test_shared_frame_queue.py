"""Shared buffers must preserve frame identity across reuse and Windows spawn."""
import multiprocessing as mp
from queue import Full
import unittest

import numpy as np

from common.messages import FrameMessage
from common.shared_frame_queue import SharedFrameQueue


def send_frames(queue):
    pixels = np.empty((24, 32, 3), np.uint8)
    for index in range(40):
        pixels.fill(index)
        queue.put(FrameMessage(index, index/30, pixels), timeout=5)
        # A normal Queue's feeder could pickle this later. SharedFrameQueue
        # must already own a snapshot before put returns.
        pixels.fill(255)
    queue.put(None, timeout=5)


class SharedFrameQueueTests(unittest.TestCase):
    def test_spawn_reused_producer_array_keeps_exact_ids_and_pixels(self):
        context = mp.get_context('spawn')
        queue = SharedFrameQueue(context, (32, 24), maxsize=2)
        producer = context.Process(target=send_frames, args=(queue,))
        producer.start()
        snapshots = []
        try:
            for expected in range(40):
                message = queue.get(timeout=10)
                self.assertEqual(message.frame_id, expected)
                self.assertTrue(np.all(message.frame == expected))
                snapshots.append(message.frame)
            self.assertIsNone(queue.get(timeout=10))
            producer.join(5)
            self.assertEqual(producer.exitcode, 0)
            for index, pixels in enumerate(snapshots):
                self.assertTrue(np.all(pixels == index))
        finally:
            if producer.is_alive():
                producer.terminate()
                producer.join(5)
            queue.cancel_join_thread()
            queue.close()

    def test_full_and_invalid_puts_do_not_corrupt_pending_frame_or_leak_slots(self):
        queue = SharedFrameQueue(mp.get_context('spawn'), (32, 24))
        try:
            first = FrameMessage(1, 0., np.full((24, 32, 3), 7, np.uint8))
            queue.put(first)
            for _ in range(10):
                with self.assertRaises(Full):
                    queue.put_nowait(FrameMessage(2, 0., first.frame+1))
            saved = queue.get(timeout=5)
            self.assertEqual(saved.frame_id, 1)
            np.testing.assert_array_equal(saved.frame, first.frame)
            with self.assertRaises(ValueError):
                queue.put_nowait(FrameMessage(3, 0., np.zeros((5, 5, 3), np.uint8)))
            queue.put_nowait(first)
            self.assertEqual(queue.get(timeout=5).frame_id, 1)
            queue.put(None)
            self.assertIsNone(queue.get(timeout=5))
        finally:
            queue.cancel_join_thread()
            queue.close()
