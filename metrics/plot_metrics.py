import csv
from pathlib import Path


def plot_metric(csv_path, field_name):
    import matplotlib.pyplot as plt
    timestamps = []
    values = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            value = row.get(field_name)
            if value in (None, "", "None"):
                continue

            timestamps.append(float(row["timestamp"]))
            values.append(float(value))

    if not timestamps:
        return

    t0 = timestamps[0]
    relative_time = [t - t0 for t in timestamps]

    plt.figure()
    plt.plot(relative_time, values)
    plt.xlabel("Time [s]")
    plt.ylabel(field_name)
    plt.title(f"{field_name} over time")
    plt.tight_layout()
    plt.show()


def save_runtime_plot(csv_path, output_path, pose_limit=None):
    """Save measured latency and fit quality; missing poses remain gaps."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    with open(csv_path,newline='',encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return False
    def values(key):
        return np.array([float(r[key]) if r.get(key) not in (None,'','None') else np.nan for r in rows])
    time = values('timestamp')
    fig, axes = plt.subplots(3,1,figsize=(11,8),sharex=True,layout='constrained')
    axes[0].plot(time,values('processing_time_ms'),label='Fast processing',lw=1)
    axes[0].plot(time,values('latency_ms'),label='Frame to result',lw=1,alpha=.7)
    axes[0].set_ylabel('Time (ms)')
    axes[0].legend()
    axes[1].plot(time,values('reprojection_error_px'),lw=1)
    if pose_limit is not None:
        axes[1].axhline(pose_limit,color='red',ls='--',label='Pose acceptance limit')
        axes[1].legend()
    axes[1].set_ylabel('RMS (working px)')
    available = [int(r.get('pose_valid')=='True') for r in rows]
    axes[2].step(time,available,where='post',lw=1)
    axes[2].set_yticks([0,1],['Unavailable','Available'])
    axes[2].set_ylabel('Pose output')
    axes[2].set_xlabel('Source video time (s)')
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle('Measured runtime behaviour — availability is not ground-truth accuracy')
    Path(output_path).parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output_path,dpi=150)
    plt.close(fig)
    return True


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',type=Path,default=Path('outputs/runtime/metrics.csv'))
    parser.add_argument('--output',type=Path,default=Path('outputs/runtime/performance.png'))
    parser.add_argument('--pose-limit',type=float)
    args = parser.parse_args()
    save_runtime_plot(args.csv,args.output,args.pose_limit)
