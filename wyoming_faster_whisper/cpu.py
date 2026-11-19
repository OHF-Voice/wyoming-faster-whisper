"""The default for --cpu-threads: one thread per physical core we may run on.

Inference here is matrix multiplication, which saturates a core's execution
units on its own, so a second hyperthread on the same core adds contention
rather than throughput. Measured on an 8-core/16-thread Ryzen 7 5700X, every
backend was fastest at 8 threads and slower at 16 than at the old default of
4 (ms per command):

    threads   qwen3-asr   sherpa (Parakeet)   faster-whisper (medium-int8)
          4         478                 146                           2877
          8         402                 133                           2316
         16         518                 234                           2648

onnxruntime makes the same choice for its own default.
"""

import os
from pathlib import Path


def default_cpu_threads(root: Path = Path("/")) -> int:
    """Return the number of physical cores this process may run on.

    CPU affinity (taskset, docker --cpuset-cpus) and a cgroup v2 CPU quota on
    the container (docker --cpus) narrow it. Off Linux every logical CPU counts.
    ``root`` is where /sys is looked up, for tests.
    """
    if hasattr(os, "sched_getaffinity"):
        cpus = os.sched_getaffinity(0)
    else:
        cpus = set(range(os.cpu_count() or 1))

    cpu_dir = root / "sys" / "devices" / "system" / "cpu"
    try:
        # Hyperthreads of one core share a siblings list.
        threads = len(
            {
                (cpu_dir / f"cpu{cpu}" / "topology" / "thread_siblings_list")
                .read_text(encoding="utf-8")
                .strip()
                for cpu in cpus
            }
        )
    except OSError:
        threads = len(cpus)

    try:
        quota, period = (
            (root / "sys" / "fs" / "cgroup" / "cpu.max")
            .read_text(encoding="utf-8")
            .split()
        )
        if quota != "max":
            threads = min(threads, int(quota) // int(period))
    except (OSError, ValueError):
        pass

    return max(1, threads)
