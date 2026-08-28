import json
import time
from contextlib import contextmanager

import torch


class PerformanceMonitor:
    """Track elapsed time and peak CUDA memory for named experiment phases."""

    def __init__(self, device=None):
        self.device = device
        self.metrics = {}

    @staticmethod
    def _empty_phase_metrics():
        return {
            "times": [],
            "peak_gpu_memory_gb": [],
        }

    def _cuda_device(self):
        if not torch.cuda.is_available():
            return None

        if self.device is None:
            return torch.cuda.current_device()

        device = torch.device(self.device)
        if device.type != "cuda":
            return None

        return device

    @contextmanager
    def monitor(self, phase):
        self.metrics.setdefault(phase, self._empty_phase_metrics())

        cuda_device = self._cuda_device()
        if cuda_device is not None:
            torch.cuda.reset_peak_memory_stats(cuda_device)
            torch.cuda.synchronize(cuda_device)

        start_time = time.perf_counter()

        try:
            yield
        finally:
            if cuda_device is not None:
                torch.cuda.synchronize(cuda_device)

            elapsed_time = time.perf_counter() - start_time
            peak_memory_gb = 0.0

            if cuda_device is not None:
                peak_memory_gb = torch.cuda.max_memory_allocated(cuda_device) / 1024**3

            self.metrics[phase]["times"].append(elapsed_time)
            self.metrics[phase]["peak_gpu_memory_gb"].append(peak_memory_gb)

    def get_snapshot(self, phase):
        self.metrics.setdefault(phase, self._empty_phase_metrics())
        return len(self.metrics[phase]["times"])

    def record(self, phase, elapsed_time, peak_gpu_memory_gb=0.0):
        self.metrics.setdefault(phase, self._empty_phase_metrics())
        self.metrics[phase]["times"].append(float(elapsed_time))
        self.metrics[phase]["peak_gpu_memory_gb"].append(float(peak_gpu_memory_gb))

    def summarize_phase(self, phase, start_idx=0):
        if phase not in self.metrics:
            return None

        times = self.metrics[phase]["times"][start_idx:]
        peak_memory = self.metrics[phase]["peak_gpu_memory_gb"][start_idx:]

        if not times:
            return None

        return {
            "total_time": float(sum(times)),
            "avg_time": float(sum(times) / len(times)),
            "peak_gpu_memory_gb": float(max(peak_memory)) if peak_memory else 0.0,
            "num_sessions": len(times),
        }

    def get_all_stats(self):
        return {
            phase: self.summarize_phase(phase)
            for phase in self.metrics
        }

    # Compatibility helpers used by GLAM's phase-level instrumentation.
    def reset_metrics(self):
        self.metrics = {}

    def monitor_training(self):
        return self.monitor("training")

    def monitor_inference(self):
        return self.monitor("inference")

    def get_training_snapshot(self):
        return self.get_snapshot("training")

    def get_inference_snapshot(self):
        return self.get_snapshot("inference")

    def summarize_training_since(self, start_idx):
        summary = self.summarize_phase("training", start_idx)
        if summary is None:
            return None
        return {
            "training_time": summary["total_time"],
            "avg_training_time": summary["avg_time"],
            "gpu_memory_gb": summary["peak_gpu_memory_gb"],
            "gpu_utilization_pct": 0.0,
            "num_training_sessions": summary["num_sessions"],
        }

    def summarize_inference_since(self, start_idx):
        summary = self.summarize_phase("inference", start_idx)
        if summary is None:
            return None
        return {
            "inference_time": summary["total_time"],
            "avg_inference_time": summary["avg_time"],
            "inference_gpu_memory_gb": summary["peak_gpu_memory_gb"],
            "inference_gpu_utilization_pct": 0.0,
            "num_inference_sessions": summary["num_sessions"],
        }

    def save_metrics(self, filepath):
        with open(filepath, "w") as f:
            json.dump(self.get_all_stats(), f, indent=2)

    def print_stats(self):
        for phase, stats in self.get_all_stats().items():
            if stats is None:
                continue
            print(
                f"{phase.title()}: total={stats['total_time']:.2f}s, "
                f"average={stats['avg_time']:.2f}s, "
                f"peak GPU memory={stats['peak_gpu_memory_gb']:.2f} GB"
            )
