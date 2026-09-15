from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    # Physical A4 dimensions in meters
    a4_width_m: float = 0.210
    a4_height_m: float = 0.297

    # Slow detector
    slow_detection_interval_s: float = 1.0

    # Tracking thresholds
    min_tracking_points: int = 6
    min_ransac_inlier_ratio: float = 0.5
    max_reprojection_error_px: float = 5.0

    # Video
    video_source: str | int = 0


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime limits; search intervals use source seconds, not wall-clock time."""
    source: str | int
    output: str
    headless: bool = False
    width: int = 1280
    max_frames: int = 0
    playback_speed: float = 1.0
    history_size: int = 16       # Received frames retained for delayed-detection replay.
    max_frame_gap: int = 5       # Largest allowed source-ID step, including dropped frames.
    slow_interval_s: float = 1.0
    local_interval_s: float = .2
    pose_reset_after: int = 3    # Consecutive rejected tracked poses before resetting.
    save_every: int = 30
    opencv_threads: int = 1
