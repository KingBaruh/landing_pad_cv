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


def _prepare_camera_parameters(config, params):
    """Return original/working sizes, scaled K, distortion and a pixel error limit.

    Sizes are (width, height). The saved calibration must match the original
    capture mode; resizing does not compensate for a different lens or crop.
    """
    original_size = tuple(int(value) for value in params['image_size'])
    resize_scale = min(1.0, config.width / original_size[0])
    working_size = (
        round(original_size[0] * resize_scale),
        round(original_size[1] * resize_scale),
    )

    width_scale = working_size[0] / original_size[0]
    height_scale = working_size[1] / original_size[1]
    # Focal lengths and principal point are measured in pixels. Scale each
    # image axis separately because rounding can slightly change the ratio.
    camera_matrix = np.array(params['camera_matrix'], dtype=float, copy=True)
    camera_matrix[0] *= width_scale
    camera_matrix[1] *= height_scale
    # Distortion coefficients describe normalized camera coordinates and do
    # not scale with image dimensions.
    distortion_coefficients = np.asarray(params['dist_coeffs'], dtype=float)

    if (not np.isfinite(camera_matrix).all()
            or not np.isfinite(distortion_coefficients).all()):
        raise ValueError('Calibration contains nonfinite values.')

    # Express the 5-original-pixel gate in working pixels (1.667 at one third
    # size). The smaller axis scale keeps the scalar gate conservative.
    pose_error_limit = AppConfig().max_reprojection_error_px * min(width_scale, height_scale)
    return original_size, working_size, camera_matrix, distortion_coefficients, pose_error_limit


def _collect_worker_events(status_queue, events):
    """Record all currently available worker messages and print starts/errors."""
    while True:
        try:
            event = status_queue.get_nowait()
        except Empty:
            break

        events.append(event)
        if event['event'] == 'error':
            print(event['error'], flush=True)
        elif event['event'] == 'started':
            print(f"{event['worker']} process started (PID {event['pid']})", flush=True)


def _wait_for_workers(processes, status_queue, events, stop_event):
    """Monitor workers until completion, a stop request, or a shutdown timeout."""
    first_exit_time = None
    while any(process.is_alive() for process in processes):
        _collect_worker_events(status_queue, events)

        if any(process.exitcode not in (None, 0) for process in processes):
            stop_event.set()
        if stop_event.is_set():
            break

        if any(process.exitcode is not None for process in processes):
            first_exit_time = first_exit_time or monotonic()
            if monotonic() - first_exit_time > 20:
                raise TimeoutError('Workers did not finish after end of stream.')

        stop_event.wait(0.05)


def _stop_and_join_workers(processes, stop_event, forced_terminations):
    """Allow graceful shutdown first; terminate workers only as a fallback."""
    if any(process.is_alive() for process in processes):
        stop_event.set()

    for process in processes:
        process.join(timeout=3)

    for process in processes:
        if process.is_alive():
            forced_terminations.append(process.name)
            process.terminate()
            process.join(timeout=3)


def _write_runtime_summary(output_directory, report):
    summary_path = output_directory / 'runtime_summary.json'
    summary_path.write_text(json.dumps(report, indent=2), encoding='utf-8')


def _export_performance_graphs(output_directory, report, events, exit_code):
    """Export only when Fast confirms that this run's data has been saved."""
    fast_finished = next(
        (
            event for event in reversed(events)
            if event['worker'] == 'Fast' and event['event'] == 'finished'
        ),
        None,
    )
    can_export = (
        exit_code in (0, 130)
        and fast_finished is not None
        and fast_finished['stats']['processed_frames'] >= 2
    )

    if can_export:
        try:
            from metrics.export_report_figures import export

            print('Creating performance graphs...', flush=True)
            graph_directory = export(output_directory)
            report['graphs'] = dict(status='created', directory=str(graph_directory))
            print(f'Graphs saved: {graph_directory}', flush=True)
        except Exception as error:
            report['graphs'] = dict(status='error', reason=str(error))
            print(
                f'Graph export failed: {error}. Runtime data remains in {output_directory}',
                flush=True,
            )
            exit_code = 1
    else:
        if exit_code == 1 or fast_finished is None:
            reason = 'worker_failure_or_incomplete_output'
        else:
            reason = 'fewer_than_two_frames'
        report['graphs'] = dict(status='skipped', reason=reason)
        print(f'Graphs not generated for this run: {reason}', flush=True)

    return exit_code


def run(config, params, stop_event=None):
    """Run three workers using RuntimeConfig and loaded calibration parameters.

    Supervise shutdown and export graphs after workers save their results.
    Return 0 on success, 130 on keyboard interruption, or 1 on a reported
    worker/export failure. Setup and supervisory exceptions can propagate.
    """
    (
        original_size,
        working_size,
        camera_matrix,
        distortion_coefficients,
        pose_error_limit,
    ) = _prepare_camera_parameters(config, params)

    # Windows starts each worker in a fresh Python process.
    context = mp.get_context('spawn')
    stop_event = stop_event if stop_event is not None else context.Event()
    fast_ready = context.Event()
    slow_ready = context.Event()

    # Bound image queues to prevent an ever-growing backlog. Frame/display
    # traffic may be dropped; requests, replies and EOF use reliable delivery.
    frame_queue = context.Queue(maxsize=1)          # Video -> Fast
    detection_request_queue = context.Queue(maxsize=1)  # Fast -> Slow
    detection_result_queue = context.Queue(maxsize=1)   # Slow -> Fast
    display_result_queue = context.Queue(maxsize=1)     # Fast -> Video
    status_queue = context.Queue()                # Workers -> supervisor

    worker_definitions = [
        (
            'Fast',
            fast_process_main,
            (
                frame_queue, detection_request_queue, detection_result_queue,
                display_result_queue, camera_matrix, distortion_coefficients,
                working_size, pose_error_limit, config, stop_event, fast_ready,
            ),
        ),
        (
            'Slow',
            slow_process_main,
            (
                detection_request_queue, detection_result_queue, camera_matrix,
                pose_error_limit, config, stop_event, slow_ready,
            ),
        ),
        (
            'Video',
            video_player_main,
            (
                frame_queue, display_result_queue, original_size, working_size,
                config, stop_event, fast_ready, slow_ready,
            ),
        ),
    ]
    processes = [
        context.Process(
            name=name,
            target=worker_entry,
            args=(name, target, arguments, status_queue, stop_event),
        )
        for name, target, arguments in worker_definitions
    ]

    output_directory = Path(config.output)
    output_directory.mkdir(parents=True, exist_ok=True)
    events = []
    forced_terminations = []
    interrupted = False
    started_processes = []

    try:
        for process in processes:
            process.start()
            started_processes.append(process)
        _wait_for_workers(started_processes, status_queue, events, stop_event)
    except KeyboardInterrupt:
        interrupted = True
        stop_event.set()
    finally:
        # Normal EOF propagates through the workers' queues. Q, Ctrl+C and
        # errors use the shared stop event. Both paths must join the workers.
        _stop_and_join_workers(started_processes, stop_event, forced_terminations)
        _collect_worker_events(status_queue, events)

        report = dict(
            config=asdict(config),
            original_size=original_size,
            working_size=working_size,
            coordinate_space='undistorted_working_resolution',
            pose_error_limit_px=pose_error_limit,
            interrupted=interrupted,
            stop_requested=stop_event.is_set(),
            forced_termination=forced_terminations,
            exit_codes={process.name: process.exitcode for process in started_processes},
            events=events,
        )
        # Graph export reads this summary, so save it before exporting.
        _write_runtime_summary(output_directory, report)

        for queue in (
            frame_queue, detection_request_queue, detection_result_queue,
            display_result_queue, status_queue,
        ):
            queue.cancel_join_thread()
            queue.close()

    if forced_terminations or any(process.exitcode != 0 for process in started_processes):
        exit_code = 1
    elif interrupted:
        exit_code = 130
    else:
        exit_code = 0

    exit_code = _export_performance_graphs(output_directory, report, events, exit_code)
    _write_runtime_summary(output_directory, report)
    return exit_code


def _parse_arguments():
    """Read command-line options and validate their existing limits."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--video', type=Path)
    source.add_argument('--camera', type=int)
    parser.add_argument('--camera-params', type=Path, default=Path('calibration/camera_params.npz'))
    parser.add_argument('--output', type=Path, default=Path('outputs/runtime'))
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--playback-speed', type=float, default=1.0)
    parser.add_argument('--history-size', type=int, default=16)
    parser.add_argument('--max-frame-gap', type=int, default=5)
    parser.add_argument('--slow-interval', type=float, default=1.0)
    parser.add_argument('--local-interval', type=float, default=0.2)
    parser.add_argument('--pose-reset-after', type=int, default=3)
    parser.add_argument('--save-every', type=int, default=30)
    args = parser.parse_args()

    if (
        args.width < 100
        or args.max_frames < 0
        or min(args.history_size, args.max_frame_gap, args.pose_reset_after, args.save_every) < 1
        or not all(
            np.isfinite(value) and value > 0
            for value in (args.playback_speed, args.slow_interval, args.local_interval)
        )
    ):
        parser.error('Frame limits, dimensions, speed and intervals must be positive.')
    if args.video is not None and not args.video.is_file():
        parser.error(f'Video not found: {args.video}')

    return parser, args


def main():
    parser, args = _parse_arguments()

    camera_params = load_camera_params(str(args.camera_params))
    required_fields = ('camera_matrix', 'dist_coeffs', 'image_size')
    if camera_params is None or any(camera_params.get(key) is None for key in required_fields):
        parser.error('Calibration must contain camera_matrix, dist_coeffs and image_size.')

    config = RuntimeConfig(
        source=str(args.video) if args.video is not None else args.camera,
        output=str(args.output),
        headless=args.headless,
        width=args.width,
        max_frames=args.max_frames,
        playback_speed=args.playback_speed,
        history_size=args.history_size,
        max_frame_gap=args.max_frame_gap,
        slow_interval_s=args.slow_interval,
        local_interval_s=args.local_interval,
        pose_reset_after=args.pose_reset_after,
        save_every=args.save_every,
    )
    exit_code = run(config, camera_params)
    print(f'Runtime finished. Results: {config.output}', flush=True)
    return exit_code


if __name__ == '__main__':
    mp.freeze_support()
    raise SystemExit(main())
