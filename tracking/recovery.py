"""Low-rate redetection near the last position, with a full-frame fallback."""
import numpy as np

from detection.slow_detector import DetectionOutput


class RecoverySchedule:
    """Retry a recent local position between the less frequent full searches."""

    def __init__(self, full_interval=30, local_interval=5):
        if min(full_interval, local_interval) < 1:
            raise ValueError('Recovery intervals must be positive.')
        self.full_interval = full_interval
        self.local_interval = local_interval
        self.next_full = 0
        self.next_local = 0

    def due(self, frame_id, last_valid_frame=None):
        if frame_id >= self.next_full:
            return 'full'
        if (last_valid_frame is not None and
                0 <= frame_id-last_valid_frame <= self.full_interval and
                frame_id >= self.next_local):
            return 'local'
        return None

    def attempted(self, frame_id, mode):
        self.next_local = frame_id+self.local_interval
        if mode == 'full':
            self.next_full = frame_id+self.full_interval


def detect_for_recovery(frame, detector, last_corners=None, *, allow_global=True):
    """Return (detection, search_scope, detector_calls).

    All returned corners are in full-frame pixels. A stale local window never
    prevents a scheduled full-frame search, and the normal quad/X checks apply
    to both. allow_global=False is a local retry between full search deadlines.
    """
    height, width = frame.shape[:2]
    calls = 0
    if last_corners is not None:
        corners = np.asarray(last_corners, dtype=np.float32).reshape(4, 2)
        if np.isfinite(corners).all():
            lo, hi = corners.min(axis=0), corners.max(axis=0)
            # An early retry uses a tighter window around a recent observation.
            # Scheduled searches keep the wider window and full-frame fallback.
            margin = np.maximum((hi-lo)*(.65 if allow_global else .25), 24)
            x0, y0 = np.floor(np.maximum(lo-margin, 0)).astype(int)
            x1, y1 = np.ceil(np.minimum(hi+margin, [width, height])).astype(int)
            if x1-x0 >= 24 and y1-y0 >= 24 and (x1-x0)*(y1-y0) < .9*width*height:
                options = {'pixel_offset': (x0,y0)} if detector.camera_matrix is not None else {}
                result = detector.detect(frame[y0:y1, x0:x1], **options)
                calls += 1
                if result.valid:
                    return DetectionOutput(True, result.corners+np.float32([x0, y0]),
                                           result.confidence), 'local', calls
    if not allow_global:
        return DetectionOutput(False, None, 0.), 'local' if calls else 'none', calls
    result = detector.detect(frame)
    return result, 'global', calls+1
