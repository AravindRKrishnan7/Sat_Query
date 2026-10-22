#!/usr/bin/env python3
"""Report the local runtime relevant to SatQuery inference.

The script has no third-party dependencies and exits successfully when PyTorch
is not installed. A present but broken PyTorch installation is reported as an
error and returns a non-zero status.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


def format_bytes(value: Optional[int]) -> str:
    if value is None:
        return "unknown"
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return "unknown"


def run_text(command: list[str]) -> Optional[str]:
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return result.stdout


def total_memory_bytes() -> Optional[int]:
    if sys.platform == "darwin":
        value = run_text(["sysctl", "-n", "hw.memsize"])
        if value and value.strip().isdigit():
            return int(value.strip())
        profile = run_text(["system_profiler", "SPHardwareDataType"])
        match = re.search(r"^\s*Memory:\s*([\d.]+)\s*(GB|MB)", profile or "", re.M)
        if match:
            multiplier = 1024**3 if match.group(2) == "GB" else 1024**2
            return int(float(match.group(1)) * multiplier)
    elif sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            pass
    return None


def macos_free_memory_percent() -> str:
    if sys.platform != "darwin":
        return "not reported"
    output = run_text(["memory_pressure", "-Q"])
    match = re.search(r"free percentage:\s*(\d+)%", output or "")
    return f"{match.group(1)}%" if match else "unknown"


def print_environment() -> None:
    disk = shutil.disk_usage(Path.cwd())
    print("SatQuery environment")
    print(f"Python: {platform.python_implementation()} {platform.python_version()}")
    print(f"Executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    print(f"Machine: {platform.machine()}")
    print(f"Logical CPUs: {os.cpu_count() or 'unknown'}")
    print(f"Physical memory: {format_bytes(total_memory_bytes())}")
    if sys.platform == "darwin":
        print(f"System-wide memory free: {macos_free_memory_percent()}")
    print(f"Disk total ({Path.cwd()}): {format_bytes(disk.total)}")
    print(f"Disk free ({Path.cwd()}): {format_bytes(disk.free)}")


def print_torch_status() -> int:
    print("\nPyTorch")
    if importlib.util.find_spec("torch") is None:
        print("Installed: no")
        print("CPU test: not run")
        print("MPS built: unknown")
        print("MPS available: unknown")
        return 0

    try:
        import torch
    except Exception as exc:  # A broken binary install should be visible.
        print("Installed: present but import failed")
        print(f"Import error: {type(exc).__name__}: {exc}")
        return 1

    print("Installed: yes")
    print(f"Version: {torch.__version__}")
    try:
        result = (torch.ones(1, device="cpu") + 1).item()
        print(f"CPU test: {'available' if result == 2 else 'unexpected result'}")
    except Exception as exc:
        print(f"CPU test: failed ({type(exc).__name__}: {exc})")

    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is None:
        print("MPS built: false")
        print("MPS available: false")
    else:
        print(f"MPS built: {str(bool(mps.is_built())).lower()}")
        print(f"MPS available: {str(bool(mps.is_available())).lower()}")

    cuda = getattr(torch, "cuda", None)
    print(f"CUDA available: {str(bool(cuda and cuda.is_available())).lower()}")
    return 0


def main() -> int:
    print_environment()
    return print_torch_status()


if __name__ == "__main__":
    raise SystemExit(main())
