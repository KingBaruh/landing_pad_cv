"""Summarize frozen-run results and explicitly bounded manual visual review."""
import hashlib
import json
import statistics
from pathlib import Path

out = Path('outputs/heldout_test/landing_pad_test')
data = json.loads((out / 'results.json').read_text(encoding='utf-8'))
hashes = json.loads((out / 'evaluated_files_sha256.json').read_text(encoding='utf-8'))
for name, expected in hashes.items():
    with Path(name).open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected, name

frames = data['frames']
def intervals(predicate):
    result = []
    for frame in frames:
        if predicate(frame):
            i = frame['frame_id']
            if result and result[-1][1] == i - 1:
                result[-1][1] = i
            else:
                result.append([i, i])
    return result

errors = [f['pose']['reprojection_error_px'] for f in frames if (f.get('pose') or {}).get('valid')]
rejected = intervals(lambda f: f['valid'] and not (f.get('pose') or {}).get('valid'))
reviewed_absence = [[380, 450], [720, 755]]
for start, end in reviewed_absence:
    assert all(not f['valid'] and not (f.get('pose') or {}).get('valid')
               for f in frames if start <= f['frame_id'] <= end)
summary = {
    'source': data['source'],
    'evaluation': 'New video evaluated without changing algorithm or thresholds',
    'evaluated_file_hashes_unchanged': True,
    'metadata_frame_count': 1008,
    'decoded_frames': data['processed_frames'],
    'tracking_valid_frames': data['valid_frames'],
    'pose_valid_frames': len(errors),
    'pose_reprojection_rms_px': {'minimum': min(errors), 'median': statistics.median(errors), 'maximum': max(errors)},
    'tracking_valid_pose_rejected_frames': sum(b-a+1 for a,b in rejected),
    'tracking_valid_pose_rejected_intervals_inclusive': rejected,
    'tracking_invalid_intervals_inclusive': intervals(lambda f: not f['valid']),
    'visual_review': {
        'method': 'Every 30th decoded frame, plus every 5th frame around exits and returns; not exhaustive ground-truth annotation',
        'conservative_target_absence_intervals_inclusive': reviewed_absence,
        'tracking_or_pose_outputs_in_these_intervals': 0,
        'returns': [
            {'fully_visible_by_frame': 480, 'previous_sample_still_partial_frame': 475, 'recovered_frame': 497, 'approx_recovery_seconds': [0.57, 0.70]},
            {'fully_visible_by_frame': 775, 'previous_sample_still_partial_frame': 770, 'recovered_frame': 803, 'approx_recovery_seconds': [0.93, 1.07]},
        ],
    },
    'limitations': [
        'Pose-valid fraction of all frames is availability, not detection accuracy: the video includes target absence and clipping.',
        'Low reprojection error is an internal fit check, not independent proof of metric distance accuracy.',
        'Physical distance and orientation were not measured.',
        'This video is a new take on a familiar tabletop, not broad background/camera generalization.',
        'If used to tune the algorithm next, this video becomes development data and a further unseen video is needed.',
    ],
}
(out / 'validation.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
report = f'''# New-video evaluation: landing_pad_test.mp4

The existing algorithm and thresholds were used unchanged. Source, camera parameters,
and input video SHA-256 hashes are in `evaluated_files_sha256.json`; they were
verified again when this report was generated.

Command:
```
python -m tracking.demo --video videos/landing_pad_test.mp4 --camera-params calibration/camera_params.npz --pose --output outputs/heldout_test
```

## Results

- 1,006 sequentially decoded frames; container metadata reports 1,008.
- Tracking reported valid in 691 frames; Pose passed its unchanged 5 px RMS threshold in 606.
- Accepted Pose reprojection RMS: median {statistics.median(errors):.3f} px, range {min(errors):.3f}–{max(errors):.3f} px.
- 85 frames retained tracking but rejected Pose: 61–92, 94–96, 529–578 (inclusive).
- Tracking invalid: 348–496, 579–583, 676–802, 813–817, 904–932.
  These spans include intentional target exits, as well as visible-target gaps.

## Absence and recovery review

Visual review used every 30th frame and additional samples every 5 frames near
the exits and returns. Contact sheets are `presence_page_*.png` and
`transitions_*.png`. This is sampled review, not exhaustive ground truth.

The conservative reviewed absence spans, frames 380–450 and 720–755, contained
no reported tracking or Pose outputs. Full re-entry was visible by frames 480
and 775; the preceding samples (475 and 770) were still clipped. Recovery occurred
at frames 497 and 803: approximately 0.57–0.70 seconds and 0.93–1.07 seconds after
full re-entry. These timings are bounded estimates from the sampling interval.

The visible-target Pose rejection spans remain unresolved. Example previews
`frame_000090.jpg` and `frame_000540.jpg` show rejected reprojection fits.
Passing a fit threshold does not establish independently correct corner positions.

## Interpretation and remaining work

The new take supports functioning loss and reacquisition on this tabletop, with
remaining Pose failures while tracking continues. The 606/1,006 fraction is output
availability, not accuracy, because some frames intentionally lack a full target.
No physical distance or orientation reference was recorded, so metric accuracy
remains unvalidated. A new take on a familiar tabletop is a limited independent
check, not broad scene generalization.

Preserve this baseline before investigating sustained Pose rejection and recovery.
If this video is then used for tuning, use a further unseen video for final validation.

Machine-readable details: `validation.json`. Per-frame results: `results.json`.
'''
(out / 'README.md').write_text(report, encoding='utf-8')
print(json.dumps(summary, indent=2))
