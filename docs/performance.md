# Fast runtime performance

Nominal 30 FPS was observed with the prepared 640-pixel configuration. **Stable
30 FPS at the original 1280-pixel setting has not been established.** The lower
resolution also reduces accepted Pose coverage, so this is an explicit tradeoff.

The target is a nominal 30 frames per second. Fast compute capacity, observed
stream throughput, and whole-run throughput are different measurements. A
short run at 30 FPS is not a guarantee that every frame finishes within 33.33 ms.

## Run the throughput-oriented configuration

```powershell
python main.py --video videos/v1.mp4 --prepare-video --width 640 --output outputs/fps_v1
```

Use `--width 960` for more image detail, or omit `--width` for the unchanged
1280 default. Those larger sizes have less performance headroom; repeat tests
at 960 and 1280 fell below 30 FPS on this laptop. Changing width is an explicit
accuracy/performance tradeoff, not a way to claim unchanged image detail.

The first `--prepare-video` run decodes every readable source frame, applies the
same INTER_AREA resize as the runtime, and writes a lossless HuffYUV AVI plus a
manifest under `outputs/video_cache/`. Subsequent runs reuse it. The source hash,
dimensions and resize version identify the cache. Preparation is offline and
is excluded from runtime FPS. This is not a solution for live 4K decoding.
The cache preserves all decoded frames and their nominal source FPS; it does
not decimate frames or slow playback. Lossless caches can be much larger than
the phone's compressed recording and are intentionally ignored by Git.

Original calibration dimensions are checked during preparation. The runtime
scales both rows of K and the original-pixel error threshold to the selected
working size. Physical A4 dimensions, camera distortion and field of view stay
unchanged. For the 3840x2160 calibration, the Pose gate is 1.667 px at width 1280,
1.25 px at 960, and 0.833 px at 640. No Pose acceptance gate was relaxed.

## Implementation changes

- LK builds forward/backward pyramids around the tracked points with an aligned
  160-pixel halo. Harris feature search uses the paper bounds with an 8-pixel
  halo. Public corners remain in full working-image coordinates.
- X scoring evaluates the same line pairs in arrays. Robust line fitting uses
  up to 256 uniformly distributed foreground samples; all ink pixels still
  participate in the final concentration, crossing and four-arm checks.
- Image pixels use bounded shared buffers instead of being pickled through
  multiprocessing pipes. Each slot stays owned until the reader copies it;
  stored history and result images cannot turn into a different frame later.
- A six-frame input queue absorbs short spikes. It can add queueing latency;
  sustained overload still drops old input. Requests and replies remain bounded
  with one detector request outstanding, and delayed results are still replayed
  and checked before use.
- Fast reuses the current Pose already checked during delayed-result acceptance.
  It invalidates that cache on the next frame or tracking loss.
- Initial acquisition retries can occur during the first search interval without
  postponing the normal full-search deadline. The steady-state search rate and
  the geometric/X/Pose validity checks are retained.
- Fast uses two OpenCV threads by default; Video and Slow use one each. Decoder
  parallelism can be configured separately. GUI events use `pollKey` because the
  source pacing loop already handles waiting. Headless mode draws only saved
  previews.

The GUI and decoder APIs are documented in the official OpenCV
[HighGUI reference](https://docs.opencv.org/4.13.0/d7/dfc/group__highgui.html) and
[video I/O properties](https://docs.opencv.org/4.10.0/d4/d15/group__videoio__flags__base.html).

## Measurement definitions

`results.json` contains `performance`:

- `mean_processing_ms`, `p95_processing_ms`: Fast's measured per-frame work,
  including remap, tracking, delayed correction, Pose and request scheduling.
- `compute_capacity_fps`: 1000 / mean processing time. This is not end-to-end FPS.
- `stream_fps`: completed-frame intervals divided by elapsed wall time between
  the first and last normal frame completion. Startup and finalization are excluded.
- `total_fps`: processed frames / Fast elapsed wall time, including startup wait
  and waiting for the final Slow reply, before writing reports.
- `frames_over_budget`: actual frames exceeding 1000/30 ms; spikes are reported,
  not hidden by rounding the mean.

Video's summary reports mean decoding/resizing time and decoded frame count.
Check these alongside processed-frame IDs, skipped frames, latency and valid
Pose count. Pose validity is an internal geometric acceptance measure, not
ground-truth accuracy. Display can discard old results separately from tracking.

## Validation

Selected display-enabled runs at normal source speed:

| Run | Width | Stream FPS | Processed / decoded | Valid Pose | Fast mean / p95 (ms) |
|---|---:|---:|---:|---:|---:|
| v1 | 640 | 29.98 | 309 / 309 | 252 | 13.2 / 31.9 |
| v3 | 640 | 30.01 | 408 / 408 | 303 | 11.2 / 24.0 |
| v1, earlier run | 960 | 29.98 | 309 / 309 | 273 | 19.5 / 46.4 |
| v1, slower repeat | 960 | 25.26 | 266 / 309 | 221 | 33.9 / 80.7 |
| v3 | 960 | 30.02 | 408 / 408 | 398 | 19.3 / 35.6 |

The source nominal rate is approximately 30.013 FPS. Do not interpret 29.98 as
a hard mathematical guarantee of at least 30.000 FPS, or replace stream FPS
with compute capacity. In particular, the slower repeat above failed the target.
The 640-pixel v3 run has 303 accepted poses versus 398 at 960; the difference
must be considered when choosing a submission configuration.

The recorded measurements and exact configurations are in
[`performance_results.json`](performance_results.json). They include successful
runs and a slower repeat, so the results do not imply an unconditional 30 FPS
guarantee. Tests were run on Windows, an Intel i7-1165G7, Python 3.14 and OpenCV 5.0.

55 targeted tests passed: marker/distractor classification, bounded line fitting,
corner identity near crop/image boundaries, motion and loss, Pose recovery,
lossless preparation, shared-buffer ownership, real spawned-worker recovery,
EOF with detection in flight, cancellation and failure shutdown. Calibration
tests were not rerun because calibration was not modified.

The v1 width-960 vs width-1280 comparison had 268 common valid poses: median
distance difference 0.46%, 95th percentile 1.29%. This compares two estimates;
it does not replace validation against a physically measured distance.

Developer profiling tools:

```powershell
python benchmarks/profile_runtime.py --video videos/v3.mp4 --prepare-video --width 640 --headless --output outputs/profile
python benchmarks/benchmark_fast.py --video videos/v3.mp4 --cache outputs/fps_review/v3.npy --output outputs/fps_review/compute.json --threads 2
```

Profiling adds overhead and is not used for final FPS claims. The second tool
processes cached frames without dropping them and uses a fixed six-frame Slow
delay; it measures Fast computation, not live end-to-end throughput. Use a fresh
cache path for a different video or working size.
