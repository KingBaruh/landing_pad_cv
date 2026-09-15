import cv2
import numpy as np


def draw_fast_result(frame, result):
    if result is None:
        return frame

    status = "Valid" if result.pose_valid else "Invalid"

    # A compact solid panel keeps numeric output readable on white paper too.
    cv2.rectangle(frame,(4,4),(min(frame.shape[1]-1,760),210),(25,25,25),-1)

    cv2.putText(
        frame,
        f"{status} | {result.state} | frame={result.frame_id} | age={result.latency_ms:.0f} ms",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if result.corners is not None:
        pts = np.asarray(result.corners, dtype=np.int32).reshape(-1, 1, 2)
        color = (0,220,0) if result.pose_valid else (0,180,255)
        cv2.polylines(frame, [pts], True, color, 2)

        for p in pts.reshape(-1, 2):
            cv2.circle(frame, tuple(p), 5, color, -1)

        # The physical rectangle centre projects to the diagonal intersection,
        # not the arithmetic mean of image vertices under perspective.
        unit = np.float32([[0,0],[1,0],[1,1],[0,1]])
        H = cv2.getPerspectiveTransform(unit,np.asarray(result.corners,dtype=np.float32))
        center = cv2.perspectiveTransform(np.float32([[[.5,.5]]]),H)[0,0]
        cv2.circle(frame, tuple(np.round(center).astype(int)), 6, color, -1)

    y = 60
    xyz = 'N/A' if result.position_xyz is None else ', '.join(f'{v:.3f}' for v in result.position_xyz)
    rpy = 'N/A' if result.orientation_rpy is None else ', '.join(f'{v:.1f}' for v in np.rad2deg(result.orientation_rpy))
    distance = 'N/A' if result.distance_m is None else f'{result.distance_m:.3f}'
    lines = [
        f"Confidence: {result.confidence:.2f}",
        f"Distance [m]: {distance}",
        f"Position [m]: {xyz}",
        f"RPY [deg]: {rpy} | yaw mod 180",
        result.reason,
    ]

    for line in lines:
        cv2.putText(
            frame,
            str(line),
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 28

    return frame
