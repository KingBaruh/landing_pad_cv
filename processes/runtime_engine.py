"""Fast-process state, including temporal alignment of asynchronous detections."""
from collections import OrderedDict
from time import monotonic, perf_counter

from common.messages import DetectionRequest, FastResult
from geometry.pose import estimate_pose
from tracking.tracker import LandingPadTracker


class FastEngine:
    """Own tracking state and delayed-detection history inside the Fast worker.

    Frames, corners, K and pose_limit share undistorted working-image pixels.
    Source timestamps schedule searches; wall-clock timers measure their cost.
    """

    def __init__(self, K, pose_limit, config):
        self.K, self.pose_limit, self.config = K, pose_limit, config
        self.tracker = LandingPadTracker()
        self.history = OrderedDict()
        self.pending_id = None
        self.next_full = self.next_local = 0.0
        self.last_corners = None
        self.last_valid_time = None
        self.previous_pose = None
        self.rejection_streak = 0
        self.reason = 'waiting_for_detection'
        self.state = 'SEARCHING'
        self.confidence = 0.0
        self.points = 0
        self.detection_frame_id = None
        self.replayed_frames = 0
        self.stats = dict(requests=0, accepted_detections=0, rejected_delayed_detections=0,
                          losses=0, pose_resets=0, skipped_source_frames=0)
        self.detection_events = []

    def lose(self, reason):
        """Clear the active track and Pose prior while keeping the last search ROI."""
        if self.tracker.initialized:
            self.stats['losses'] += 1
        self.tracker.reset()
        self.previous_pose = None
        self.rejection_streak = 0
        self.state, self.reason = 'LOST', reason
        self.confidence, self.points = 0.0, 0

    def advance(self, msg):
        """Store the next undistorted frame and advance an existing track.

        Reject non-increasing source IDs. Tracking failures clear the active
        state; this method does not run detection or estimate the current Pose.
        """
        if self.history:
            gap = msg.frame_id-next(reversed(self.history))
            if gap <= 0:
                raise ValueError('Fast frames must have strictly increasing IDs.')
            self.stats['skipped_source_frames'] += gap-1
            # LK assumes limited motion between observations. A large source
            # gap can invalidate that assumption even if the queue is short.
            if gap > self.config.max_frame_gap:
                self.lose('frame_gap_too_large')
        self.history[msg.frame_id] = msg
        while len(self.history) > self.config.history_size:
            self.history.popitem(last=False)
        if self.tracker.initialized:
            tracked = self.tracker.update(msg.frame)
            if tracked.valid:
                self.state, self.reason = 'TRACKING', 'tracked'
                self.confidence = tracked.inlier_ratio
                self.points = tracked.num_points
            else:
                self.stats['losses'] += 1
                self.lose(tracked.reason)

    def request(self):
        """Return a due DetectionRequest, or None; sent() records actual delivery."""
        # One outstanding request bounds Slow's backlog and makes reply/source
        # matching explicit, even while Fast continues receiving newer frames.
        if self.pending_id is not None or not self.history:
            return None
        msg = next(reversed(self.history.values()))
        # These deadlines use source seconds, so playback speed does not
        # silently change which parts of a video get scheduled for detection.
        full = msg.timestamp >= self.next_full
        recent = (self.last_valid_time is not None and
                  0 <= msg.timestamp-self.last_valid_time <= self.config.slow_interval_s)
        local = (not self.tracker.initialized and recent and msg.timestamp >= self.next_local)
        if not full and not local:
            return None
        return DetectionRequest(msg.frame_id, msg.timestamp, msg.frame, self.last_corners,
                                full, self.state, self.reason)

    def sent(self, request):
        """Advance search deadlines only after the request has entered its queue."""
        self.pending_id = request.frame_id
        self.next_local = request.timestamp+self.config.local_interval_s
        if request.allow_global:
            self.next_full = request.timestamp+self.config.slow_interval_s
        self.stats['requests'] += 1

    def accept(self, detection):
        """Initialize on the source frame, replay received frames, then validate.

        Negative/stale/unusable slow results never overwrite a healthy track.
        The temporary tracker is installed only after successful catch-up/Pose.
        Return True only when that replacement is installed; log other replies
        with their rejection reason and return False.
        """
        start = perf_counter()
        event = dict(source_frame=detection.frame_id,
                     current_frame=next(reversed(self.history)) if self.history else None,
                     scope=detection.scope, detector_ms=detection.processing_ms,
                     accepted=False, reason='', replayed_frames=0)
        def reject(reason):
            event['reason'] = reason
            event['catchup_ms'] = (perf_counter()-start)*1000
            self.detection_events.append(event)
            self.stats['rejected_delayed_detections'] += 1
            return False
        if detection.frame_id != self.pending_id:
            return reject('unexpected_request_id')
        # A matching negative reply also completes the request. Otherwise a
        # failed detection would leave future searches blocked indefinitely.
        self.pending_id = None
        if not detection.valid:
            return reject('no_detection')
        source = self.history.get(detection.frame_id)
        if source is None:
            return reject('source_frame_expired')
        if abs(source.timestamp-detection.timestamp) > 1e-6:
            return reject('timestamp_mismatch')
        # Test the correction on a separate tracker. Applying old corners
        # directly to the newest image would mix different points in time.
        candidate = LandingPadTracker()
        if not candidate.initialize(source.frame, detection.corners):
            return reject('initialization_failed')
        previous_id = source.frame_id
        for frame_id, msg in self.history.items():
            if frame_id <= source.frame_id:
                continue
            if frame_id-previous_id > self.config.max_frame_gap:
                return reject('replay_frame_gap')
            tracked = candidate.update(msg.frame)
            event['replayed_frames'] += 1
            if not tracked.valid:
                return reject('replay_'+tracked.reason)
            previous_id = frame_id
        pose = estimate_pose(candidate.corners, self.K, None, previous=self.previous_pose,
                             max_reprojection_error_px=self.pose_limit)
        if not pose.valid:
            return reject('replayed_pose_'+pose.reason)
        self.tracker = candidate
        self.previous_pose = pose
        self.rejection_streak = 0
        self.state, self.reason = 'TRACKING', 'reinitialized_from_slow'
        self.points = len(candidate.prev_points)
        self.confidence = float(detection.confidence)
        self.detection_frame_id = source.frame_id
        self.replayed_frames = event['replayed_frames']
        self.stats['accepted_detections'] += 1
        event.update(accepted=True, reason='accepted', catchup_ms=(perf_counter()-start)*1000)
        self.detection_events.append(event)
        return True

    def result(self, processing_ms=0.0, *, count_rejection=True):
        """Build a FastResult for the latest frame, with a freshly checked Pose.

        Tracking validity and Pose validity are separate. Invalid Pose fields
        contain None. At EOF, count_rejection=False avoids counting a second
        evaluation of the last frame as another consecutive failed frame.
        """
        msg = next(reversed(self.history.values()))
        valid = self.tracker.initialized
        corners = self.tracker.corners.copy() if valid else None
        pose = estimate_pose(corners, self.K, None, previous=self.previous_pose,
                             max_reprojection_error_px=self.pose_limit)
        self.previous_pose = pose if pose.valid else None
        # One bad fit can be transient. Consecutive failures trigger recovery
        # even when optical flow still reports a geometrically plausible track.
        if count_rejection:
            self.rejection_streak = self.rejection_streak+1 if valid and not pose.valid else 0
        if self.rejection_streak >= self.config.pose_reset_after:
            self.stats['pose_resets'] += 1
            self.lose('pose_inconsistent')
            valid, corners = False, None
        if valid:
            self.last_corners = corners.copy()
            self.last_valid_time = msg.timestamp
        return FastResult(msg.frame_id, msg.timestamp, valid, corners,
                          pose.distance_m if pose.valid else None,
                          pose.position_xyz if pose.valid else None,
                          pose.orientation_rpy if pose.valid else None,
                          self.confidence if valid else 0.0, self.state,
                          pose.valid, self.reason if pose.valid or not valid else pose.reason,
                          pose.reprojection_error_px, self.points if valid else 0,
                          processing_ms, (monotonic()-msg.captured_at)*1000,
                          self.detection_frame_id, self.replayed_frames, msg.frame)
