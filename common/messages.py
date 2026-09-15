"""Picklable messages. Coordinates always refer to the supplied working frame."""
from dataclasses import dataclass
import numpy as np


@dataclass
class FrameMessage:
    frame_id: int
    timestamp: float                 # source time, seconds
    frame: np.ndarray
    captured_at: float = 0.0          # monotonic wall time for latency


@dataclass
class DetectionRequest:
    frame_id: int
    timestamp: float
    frame: np.ndarray                # already undistorted; never undistort twice
    last_corners: np.ndarray | None = None
    allow_global: bool = True
    state: str = 'SEARCHING'
    reason: str = ''


@dataclass
class DetectionResult:
    frame_id: int
    timestamp: float
    valid: bool
    corners: np.ndarray | None
    confidence: float
    scope: str = 'global'
    detector_calls: int = 0
    processing_ms: float = 0.0


@dataclass
class FastResult:
    frame_id: int
    timestamp: float
    valid: bool
    corners: np.ndarray | None
    distance_m: float | None
    position_xyz: np.ndarray | None
    orientation_rpy: np.ndarray | None
    confidence: float
    state: str
    pose_valid: bool = False
    reason: str = ''
    reprojection_error_px: float | None = None
    tracked_points: int = 0
    processing_ms: float = 0.0
    latency_ms: float = 0.0
    detection_frame_id: int | None = None
    replayed_frames: int = 0
    frame: np.ndarray | None = None   # exact undistorted frame for this overlay
