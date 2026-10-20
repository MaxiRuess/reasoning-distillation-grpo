"""Low-frequency GPU sampling and stage timing; no extra package or paid work."""

import shutil
import subprocess
import threading
import time


class GpuMonitor:
    def __init__(self, interval=1):
        self.interval = interval
        self.samples = []
        self.error = None
        self.stop_event = threading.Event()
        self.thread = None
        self.cuda = False

    def __enter__(self):
        import torch

        self.start = time.monotonic()
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            torch.cuda.reset_peak_memory_stats()
            if shutil.which("nvidia-smi"):
                self.thread = threading.Thread(target=self._sample, daemon=True)
                self.thread.start()
            else:
                self.error = "nvidia-smi unavailable; utilization samples not measured"
        return self

    def _sample(self):
        query = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
            "--format=csv,noheader,nounits",
        ]
        while not self.stop_event.is_set():
            try:
                result = subprocess.run(query, capture_output=True, text=True, check=True, timeout=2)
                # These jobs request exactly one GPU. Report every visible line if the runtime exposes more.
                for line in result.stdout.strip().splitlines():
                    values = []
                    for value in line.split(","):
                        try:
                            values.append(float(value.strip()))
                        except ValueError:
                            values.append(None)
                    if len(values) == 4:
                        self.samples.append(
                            {
                                "elapsed_seconds": time.monotonic() - self.start,
                                **dict(
                                    zip(
                                        [
                                            "gpu_utilization_percent",
                                            "used_memory_mib",
                                            "total_memory_mib",
                                            "power_watts",
                                        ],
                                        values,
                                        strict=True,
                                    )
                                ),
                            }
                        )
            except (OSError, subprocess.SubprocessError) as error:
                self.error = type(error).__name__
                break
            self.stop_event.wait(self.interval)

    def __exit__(self, *args):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=3)
        self.elapsed = time.monotonic() - self.start

    def summary(self):
        import torch

        values = [
            sample["gpu_utilization_percent"]
            for sample in self.samples
            if sample["gpu_utilization_percent"] is not None
        ]
        memory = [sample["used_memory_mib"] for sample in self.samples if sample["used_memory_mib"] is not None]
        return {
            "cuda_available": self.cuda,
            "wall_seconds": self.elapsed,
            "sampling_interval_seconds": self.interval,
            "sampled_utilization_mean_percent": sum(values) / len(values) if values else None,
            "sampled_peak_device_memory_mib": max(memory) if memory else None,
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated() if self.cuda else None,
            "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved() if self.cuda else None,
            "gpu_name": torch.cuda.get_device_name() if self.cuda else None,
            "samples": self.samples,
            "sampling_error": self.error,
            "interpretation": "Device-wide sampled memory includes vLLM; torch allocator peaks cover this process only. Sampling can miss short peaks.",
        }
