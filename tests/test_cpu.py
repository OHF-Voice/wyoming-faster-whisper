"""Tests for the --cpu-threads default, against a fake /sys."""

from pathlib import Path
from typing import Dict, Optional

import pytest

from wyoming_faster_whisper import cpu

# 4 cores with 2 hyperthreads each, numbered like most x86 kernels do.
_SMT = {n: f"{n % 4},{n % 4 + 4}" for n in range(8)}


def _root(
    tmp_path: Path, siblings: Optional[Dict[int, str]], cpu_max: Optional[str] = None
) -> Path:
    """Build a fake /sys; siblings maps cpu -> thread_siblings_list."""
    for number, listing in (siblings or {}).items():
        topology = tmp_path / f"sys/devices/system/cpu/cpu{number}/topology"
        topology.mkdir(parents=True)
        (topology / "thread_siblings_list").write_text(listing + "\n")

    if cpu_max is not None:
        (tmp_path / "sys/fs/cgroup").mkdir(parents=True)
        (tmp_path / "sys/fs/cgroup/cpu.max").write_text(cpu_max + "\n")

    return tmp_path


@pytest.fixture(name="cpus")
def cpus_fixture(monkeypatch):
    """Set the logical CPUs the process may run on."""

    def set_cpus(*numbers: int) -> None:
        monkeypatch.setattr(
            cpu.os, "sched_getaffinity", lambda pid: set(numbers), raising=False
        )

    return set_cpus


def test_hyperthreads_count_once(tmp_path, cpus) -> None:
    cpus(*range(8))
    assert cpu.default_cpu_threads(_root(tmp_path, _SMT)) == 4


def test_cores_without_smt_all_count(tmp_path, cpus) -> None:
    # A Raspberry Pi 5: four cores, one thread each.
    cpus(0, 1, 2, 3)
    assert cpu.default_cpu_threads(_root(tmp_path, {n: str(n) for n in range(4)})) == 4


def test_affinity_narrows_the_cores(tmp_path, cpus) -> None:
    # Both threads of core 0, and one of core 1.
    cpus(0, 4, 1)
    assert cpu.default_cpu_threads(_root(tmp_path, _SMT)) == 2


def test_unknown_topology_counts_logical_cpus(tmp_path, cpus) -> None:
    cpus(*range(6))
    assert cpu.default_cpu_threads(_root(tmp_path, None)) == 6


@pytest.mark.parametrize(
    ("cpu_max", "expected"),
    [
        ("250000 100000", 2),  # docker --cpus 2.5: rounded down
        ("50000 100000", 1),  # never below one
        ("max 100000", 4),  # no quota
    ],
)
def test_cgroup_quota_caps_the_cores(tmp_path, cpus, cpu_max, expected) -> None:
    cpus(*range(8))
    assert cpu.default_cpu_threads(_root(tmp_path, _SMT, cpu_max)) == expected


def test_the_real_system_gets_at_least_one_thread() -> None:
    assert cpu.default_cpu_threads() >= 1
