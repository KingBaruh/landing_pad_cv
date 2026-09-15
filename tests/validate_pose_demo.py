"""Reproducible video integration check with known metric ground truth.

Run from the project root: python tests/validate_pose_demo.py
Writes only under outputs/pose_synthetic; no real calibration is overwritten.
"""
import json
from pathlib import Path
import sys
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_detection import render_scene
from test_pose import rotation
from geometry.pose import A4_OBJECT_POINTS
from tracking.demo import main


def validate():
    output = Path('outputs/pose_synthetic')
    output.mkdir(parents=True, exist_ok=True)
    video = output / 'known_pose.avi'
    calibration = output / 'camera_params.npz'
    K = np.float64([[900, 0, 640], [0, 900, 360], [0, 0, 1]])
    np.savez(calibration, camera_matrix=K, dist_coeffs=np.zeros(5), image_size=[1280, 720])
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 30, (1280, 720))
    if not writer.isOpened():
        raise RuntimeError('Cannot create the synthetic video.')
    truth = []
    try:
        for index in range(45):
            R = rotation(.3, -.35, .2+.003*index)
            xyz = np.array([.01+.0005*index, -.02, .85+.001*index])
            visible = not 12 <= index < 20
            quad = cv2.projectPoints(A4_OBJECT_POINTS, cv2.Rodrigues(R)[0], xyz, K, None)[0].reshape(4, 2)
            frame = render_scene(quad, canvas=(1280, 720)) if visible else np.full((720,1280,3), 70, np.uint8)
            writer.write(frame)
            truth.append(dict(frame_id=index, visible=visible, position_xyz_m=xyz.tolist()))
    finally:
        writer.release()
    (output / 'ground_truth.json').write_text(json.dumps(truth, indent=2), encoding='utf-8')
    with patch.object(sys, 'argv', ['tracking.demo', '--video', str(video), '--camera-params', str(calibration),
                                   '--pose', '--output', str(output), '--save-every', '10',
                                   '--redetect-every', '30', '--local-retry-every', '5']):
        main()
    report = json.loads((output / 'known_pose/results.json').read_text(encoding='utf-8'))
    errors = []
    for row, expected in zip(report['frames'], truth, strict=True):
        if not expected['visible']:
            assert not row['pose']['valid'], 'Pose persisted after disappearance.'
            assert row['pose']['position_xyz_m'] is None, 'Stale metric output.'
        if row['pose']['valid']:
            errors.append(float(np.linalg.norm(np.array(row['pose']['position_xyz_m'])-expected['position_xyz_m'])))
    assert report['processed_frames'] == 45
    assert report['pose_valid_frames'] >= 30, report['pose_valid_frames']
    assert report['initializations'] >= 2, 'Did not recover after loss.'
    assert report['full_search_attempts'] == 1, 'Unexpected full-search retry during short loss.'
    assert report['local_retry_attempts'] >= 1, 'No local recovery attempts.'
    assert max(errors) < .02, f'Metric position error: {max(errors)} m'
    summary = dict(processed_frames=45, pose_valid_frames=report['pose_valid_frames'],
                   max_position_error_m=max(errors), median_position_error_m=float(np.median(errors)),
                   initializations=report['initializations'], tracking_losses=report['tracking_losses'])
    (output / 'validation.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    validate()
