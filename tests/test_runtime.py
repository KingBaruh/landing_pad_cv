"""Temporal correctness and actual Windows-spawn process integration."""
import json
import multiprocessing as mp
from pathlib import Path
import subprocess
import sys
from time import monotonic
from threading import Timer
import unittest

import cv2
import numpy as np
from common.messages import FrameMessage, DetectionResult
from config.config import RuntimeConfig
from geometry.pose import A4_OBJECT_POINTS
from processes.runtime_engine import FastEngine
from test_detection import render_scene
from test_pose import rotation


K = np.float64([[900,0,640],[0,900,360],[0,0,1]])


def scene(index):
    xyz = np.float64([.01+.003*index,-.02,.85])
    corners = cv2.projectPoints(A4_OBJECT_POINTS,cv2.Rodrigues(rotation(.3,-.35,.2))[0],
                               xyz,K,None)[0].reshape(4,2)
    return FrameMessage(index,index/30,render_scene(corners,canvas=(1280,720)),monotonic()), corners, xyz


class AlignmentTests(unittest.TestCase):
    def engine(self,**options):
        return FastEngine(K,5,RuntimeConfig('test','outputs/test_runtime',**options))

    def test_delayed_result_replays_motion_instead_of_using_old_corners(self):
        engine = self.engine()
        source,corners,_ = scene(0)
        engine.advance(source)
        engine.sent(engine.request())
        for i in range(1,6): engine.advance(scene(i)[0])
        self.assertTrue(engine.accept(DetectionResult(0,0,True,corners,.9)))
        result = engine.result()
        expected = scene(5)
        self.assertEqual(result.frame_id,5)
        self.assertEqual(result.detection_frame_id,0)
        self.assertEqual(result.replayed_frames,5)
        # Cyclic order is normalized by the tracker.
        error = min(np.linalg.norm(result.corners-np.roll(expected[1],j,axis=0),axis=1).max() for j in range(4))
        self.assertLess(error,2.)
        self.assertLess(np.linalg.norm(result.position_xyz-expected[2]),.015)
        np.testing.assert_array_equal(result.frame,expected[0].frame)

    def test_expired_and_wrong_id_results_cannot_initialize_current_frame(self):
        engine = self.engine(history_size=2)
        source,corners,_ = scene(0)
        engine.advance(source)
        engine.sent(engine.request())
        self.assertFalse(engine.accept(DetectionResult(99,0,True,corners,.9)))
        self.assertEqual(engine.pending_id,0)
        for i in range(1,4): engine.advance(scene(i)[0])
        self.assertFalse(engine.accept(DetectionResult(0,0,True,corners,.9)))
        self.assertIsNone(engine.pending_id)
        self.assertFalse(engine.result().pose_valid)
        self.assertEqual(engine.detection_events[-1]['reason'],'source_frame_expired')

    def test_disappearance_during_replay_does_not_restore_stale_pose(self):
        engine = self.engine()
        source,corners,_ = scene(0)
        engine.advance(source)
        engine.sent(engine.request())
        engine.advance(FrameMessage(1,1/30,np.full_like(source.frame,70),monotonic()))
        self.assertFalse(engine.accept(DetectionResult(0,0,True,corners,.9)))
        result = engine.result()
        self.assertFalse(result.valid)
        self.assertIsNone(result.distance_m)

    def test_failed_periodic_detection_preserves_healthy_track(self):
        engine = self.engine(slow_interval_s=.01)
        source,corners,_ = scene(0)
        engine.advance(source)
        engine.sent(engine.request())
        self.assertTrue(engine.accept(DetectionResult(0,0,True,corners,.9)))
        engine.result()
        engine.advance(scene(1)[0])
        engine.sent(engine.request())
        self.assertFalse(engine.accept(DetectionResult(1,1/30,False,None,0)))
        self.assertTrue(engine.result().pose_valid)


class SpawnTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path('outputs/test_runtime')
        cls.root.mkdir(parents=True,exist_ok=True)
        cls.video = cls.root/'input.avi'
        cls.calibration = cls.root/'camera.npz'
        np.savez(cls.calibration,camera_matrix=K,dist_coeffs=np.zeros(5),image_size=[1280,720])
        writer = cv2.VideoWriter(str(cls.video),cv2.VideoWriter_fourcc(*'MJPG'),15,(1280,720))
        if not writer.isOpened(): raise RuntimeError('Cannot write test video.')
        try:
            for i in range(45):
                frame = scene(i//3)[0].frame
                if 16 <= i < 25: frame[:] = 70
                writer.write(frame)
        finally:
            writer.release()

    def launch(self,name,*options,calibration=None):
        output = self.root/name
        run = subprocess.run([sys.executable,'main.py','--video',str(self.video),
                              '--camera-params',str(calibration or self.calibration),
                              '--headless','--output',str(output),*options],
                             capture_output=True,text=True,timeout=40)
        report = json.loads((output/'runtime_summary.json').read_text())
        self.assertFalse(report['forced_termination'],run.stdout+run.stderr)
        self.assertTrue(all(code is not None for code in report['exit_codes'].values()))
        pids = {e['pid'] for e in report['events'] if e['event']=='started'}
        self.assertEqual(len(pids),3)
        return run,report,output

    def test_actual_processes_recover_without_stale_pose_and_exit(self):
        run,report,output = self.launch('complete')
        self.assertEqual(run.returncode,0,run.stdout+run.stderr)
        self.assertEqual(set(report['exit_codes'].values()),{0})
        self.assertEqual(report['graphs']['status'],'created')
        for name in ('00_overview','01_processing_time','02_reprojection_error','03_pose_availability'):
            for suffix in ('png','svg'):
                self.assertGreater((output/'graphs'/f'{name}.{suffix}').stat().st_size,100)
        data = json.loads((output/'results.json').read_text())
        graph_stats = json.loads((output/'graphs/statistics.json').read_text())
        self.assertEqual(graph_stats['processed_frames'],data['processed_frames'])
        self.assertEqual(graph_stats['pose_valid_frames'],data['pose_valid_frames'])
        self.assertGreater(data['pose_valid_frames'],10)
        self.assertTrue(any(r['pose_valid'] and r['frame_id'] >= 25 for r in data['frames']))
        for row in data['frames']:
            if 16 <= row['frame_id'] < 25:
                self.assertFalse(row['pose_valid'])
                self.assertIsNone(row['distance_m'])
        accepted = [e for e in data['detection_events'] if e['accepted']]
        self.assertTrue(any(e['current_frame'] > e['source_frame'] for e in accepted))

    def test_eof_with_slow_request_in_flight_finishes_last_frame(self):
        run,summary,output = self.launch('single','--max-frames','1')
        self.assertEqual(run.returncode,0,run.stdout+run.stderr)
        self.assertEqual(summary['graphs'],dict(status='skipped',reason='fewer_than_two_frames'))
        data = json.loads((output/'results.json').read_text())
        self.assertEqual(data['processed_frames'],1)
        self.assertTrue(data['frames'][0]['pose_valid'])
        self.assertEqual(data['frames'][0]['frame_id'],0)

    def test_worker_error_stops_peers_and_returns_failure(self):
        wrong = self.root/'wrong_size.npz'
        np.savez(wrong,camera_matrix=K,dist_coeffs=np.zeros(5),image_size=[640,360])
        run,report,_ = self.launch('failure',calibration=wrong)
        self.assertNotEqual(run.returncode,0)
        self.assertEqual(report['graphs']['status'],'skipped')
        self.assertTrue(any(e['event']=='error' and e['worker']=='Video' for e in report['events']))

    def test_shared_stop_event_ends_active_workers_without_termination(self):
        from main import run
        stop = mp.get_context('spawn').Event()
        config = RuntimeConfig(str(self.video),str(self.root/'cancel'),headless=True,playback_speed=.2)
        timer = Timer(1.5,stop.set)
        timer.start()
        try:
            code = run(config,dict(camera_matrix=K,dist_coeffs=np.zeros(5),image_size=[1280,720]),stop)
        finally:
            timer.cancel()
        self.assertEqual(code,0)
        report = json.loads((self.root/'cancel/runtime_summary.json').read_text())
        self.assertTrue(report['stop_requested'])
        self.assertEqual(report['graphs']['status'],'created')
        self.assertFalse(report['forced_termination'])
        self.assertEqual(set(report['exit_codes'].values()),{0})


if __name__ == '__main__':
    unittest.main()
