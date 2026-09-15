"""Three-process classical CV runtime. Run python main.py --help."""
import argparse
from dataclasses import asdict
import json
import multiprocessing as mp
from pathlib import Path
from queue import Empty
from time import monotonic

import numpy as np
from calibration.calibrate import load_camera_params
from config.config import AppConfig, RuntimeConfig
from processes.video_player import video_player_main
from processes.fast_process import fast_process_main
from processes.slow_process import slow_process_main
from processes.supervision import worker_entry


def run(config, params, stop_event=None):
    original = tuple(int(v) for v in params['image_size'])
    scale = min(1.,config.width/original[0])
    size = (round(original[0]*scale),round(original[1]*scale))
    K = np.array(params['camera_matrix'],dtype=float,copy=True)
    K[0] *= size[0]/original[0]
    K[1] *= size[1]/original[1]
    dist = np.asarray(params['dist_coeffs'],dtype=float)
    if not np.isfinite(K).all() or not np.isfinite(dist).all():
        raise ValueError('Calibration contains nonfinite values.')
    # Keep the old 5-original-pixel gate when reducing the working resolution.
    limit = AppConfig().max_reprojection_error_px*min(size[0]/original[0],size[1]/original[1])
    ctx = mp.get_context('spawn')
    stop = stop_event if stop_event is not None else ctx.Event()
    fast_ready, slow_ready = ctx.Event(),ctx.Event()
    frames, requests, detections, results = [ctx.Queue(maxsize=1) for _ in range(4)]
    status = ctx.Queue()
    jobs = [
        ('Fast',fast_process_main,(frames,requests,detections,results,K,dist,size,limit,config,stop,fast_ready)),
        ('Slow',slow_process_main,(requests,detections,K,limit,config,stop,slow_ready)),
        ('Video',video_player_main,(frames,results,original,size,config,stop,fast_ready,slow_ready)),
    ]
    processes = [ctx.Process(name=name,target=worker_entry,args=(name,target,args,status,stop))
                 for name,target,args in jobs]
    output = Path(config.output)
    output.mkdir(parents=True,exist_ok=True)
    events, forced = [], []
    interrupted = False
    def drain():
        while True:
            try:
                event = status.get_nowait()
            except Empty:
                break
            events.append(event)
            if event['event'] == 'error':
                print(event['error'],flush=True)
            elif event['event'] == 'started':
                print(f"{event['worker']} process started (PID {event['pid']})",flush=True)
    started = []
    try:
        for process in processes:
            process.start()
            started.append(process)
        first_exit = None
        while any(p.is_alive() for p in started):
            drain()
            if any(p.exitcode not in (None,0) for p in started):
                stop.set()
            if stop.is_set():
                break
            if any(p.exitcode is not None for p in started):
                first_exit = first_exit or monotonic()
                if monotonic()-first_exit > 20:
                    raise TimeoutError('Workers did not finish after end of stream.')
            stop.wait(.05)
    except KeyboardInterrupt:
        interrupted = True
        stop.set()
    finally:
        # EOF normally propagates Video -> Fast -> Slow and Fast -> Video.
        # Exceptions/Q/Ctrl+C use the shared event, then bounded joins.
        if any(p.is_alive() for p in started):
            stop.set()
        for process in started:
            process.join(timeout=3)
        for process in started:
            if process.is_alive():
                forced.append(process.name)
                process.terminate()
                process.join(timeout=3)
        drain()
        report = dict(config=asdict(config), original_size=original,working_size=size,
                      coordinate_space='undistorted_working_resolution',
                      pose_error_limit_px=limit, interrupted=interrupted,
                      stop_requested=stop.is_set(),
                      forced_termination=forced,
                      exit_codes={p.name:p.exitcode for p in started},events=events)
        (output/'runtime_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        for queue in (frames,requests,detections,results,status):
            queue.cancel_join_thread()
            queue.close()
    code = 1 if forced or any(p.exitcode != 0 for p in started) else 130 if interrupted else 0
    # Export only after workers have flushed this run's CSV/JSON and exited.
    # A successful Fast completion event prevents using leftovers from an older run.
    fast_finished = next((e for e in reversed(events)
                          if e['worker']=='Fast' and e['event']=='finished'),None)
    if code in (0,130) and fast_finished is not None and fast_finished['stats']['processed_frames'] >= 2:
        try:
            from metrics.export_report_figures import export
            print('Creating performance graphs...',flush=True)
            graph_dir = export(output)
            report['graphs'] = dict(status='created',directory=str(graph_dir))
            print(f'Graphs saved: {graph_dir}',flush=True)
        except Exception as error:
            report['graphs'] = dict(status='error',reason=str(error))
            print(f'Graph export failed: {error}. Runtime data remains in {output}',flush=True)
            code = 1
    else:
        reason = 'worker_failure_or_incomplete_output' if code == 1 or fast_finished is None else 'fewer_than_two_frames'
        report['graphs'] = dict(status='skipped',reason=reason)
        print(f'Graphs not generated for this run: {reason}',flush=True)
    (output/'runtime_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--video',type=Path)
    source.add_argument('--camera',type=int)
    parser.add_argument('--camera-params',type=Path,default=Path('calibration/camera_params.npz'))
    parser.add_argument('--output',type=Path,default=Path('outputs/runtime'))
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--width',type=int,default=1280)
    parser.add_argument('--max-frames',type=int,default=0)
    parser.add_argument('--playback-speed',type=float,default=1.)
    parser.add_argument('--history-size',type=int,default=16)
    parser.add_argument('--max-frame-gap',type=int,default=5)
    parser.add_argument('--slow-interval',type=float,default=1.)
    parser.add_argument('--local-interval',type=float,default=.2)
    parser.add_argument('--pose-reset-after',type=int,default=3)
    parser.add_argument('--save-every',type=int,default=30)
    args = parser.parse_args()
    if (args.width < 100 or args.max_frames < 0 or min(args.history_size,args.max_frame_gap,
            args.pose_reset_after,args.save_every) < 1 or
            not all(np.isfinite(v) and v > 0 for v in
                    (args.playback_speed,args.slow_interval,args.local_interval))):
        parser.error('Frame limits, dimensions, speed and intervals must be positive.')
    if args.video is not None and not args.video.is_file():
        parser.error(f'Video not found: {args.video}')
    params = load_camera_params(str(args.camera_params))
    if params is None or any(params.get(k) is None for k in ('camera_matrix','dist_coeffs','image_size')):
        parser.error('Calibration must contain camera_matrix, dist_coeffs and image_size.')
    config = RuntimeConfig(source=str(args.video) if args.video is not None else args.camera,
                           output=str(args.output),headless=args.headless,width=args.width,
                           max_frames=args.max_frames,playback_speed=args.playback_speed,
                           history_size=args.history_size,max_frame_gap=args.max_frame_gap,
                           slow_interval_s=args.slow_interval,local_interval_s=args.local_interval,
                           pose_reset_after=args.pose_reset_after,save_every=args.save_every)
    code = run(config,params)
    print(f'Runtime finished. Results: {config.output}',flush=True)
    return code


if __name__ == '__main__':
    mp.freeze_support()
    raise SystemExit(main())
