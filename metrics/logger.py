import csv
from pathlib import Path


class MetricsLogger:
    FIELDNAMES = [
        "timestamp",
        "frame_id",
        "state",
        "valid",
        "confidence",
        "tracked_points",
        "ransac_inlier_ratio",
        "reprojection_error_px",
        "processing_time_ms",
        "distance_m",
        "pose_valid",
        "latency_ms",
    ]

    def __init__(self, output_path="outputs/logs/metrics.csv"):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with self.output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writeheader()

    def log(self, **values):
        row = {key: values.get(key) for key in self.FIELDNAMES}

        with self.output_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writerow(row)
