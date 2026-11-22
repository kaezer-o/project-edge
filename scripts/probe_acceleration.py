from __future__ import annotations

import importlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    report: dict[str, object] = {
        "python": sys.version,
        "platform": sys.platform,
        "packages": {
            name: package_version(name)
            for name in ("numpy", "pandas", "torch", "numba", "cupy")
        },
        "nvidia_smi": None,
        "torch_cuda": {"installed": False},
        "numba_cuda": {"installed": False},
        "assessment": [],
    }

    executable = shutil.which("nvidia-smi")
    if executable:
        try:
            result = subprocess.run(
                [executable, "--query-gpu=name,compute_cap,driver_version,memory.total",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10, check=False,
            )
            report["nvidia_smi"] = {
                "returncode": result.returncode,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
            }
        except (OSError, subprocess.TimeoutExpired) as exc:
            report["nvidia_smi"] = {"error": str(exc)}
    else:
        report["nvidia_smi"] = {"available": False}

    if report["packages"]["torch"] is not None:
        try:
            torch = importlib.import_module("torch")
            cuda_report: dict[str, object] = {
                "installed": True,
                "torch_version": torch.__version__,
                "built_cuda_version": torch.version.cuda,
                "available": bool(torch.cuda.is_available()),
                "compiled_architectures": (
                    torch.cuda.get_arch_list() if torch.cuda.is_available() else []
                ),
            }
            if torch.cuda.is_available():
                devices = []
                for index in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(index)
                    capability = torch.cuda.get_device_capability(index)
                    devices.append({
                        "index": index,
                        "name": props.name,
                        "compute_capability": f"{capability[0]}.{capability[1]}",
                        "total_memory_bytes": props.total_memory,
                        "architecture_supported_by_build": (
                            f"sm_{capability[0]}{capability[1]}" in torch.cuda.get_arch_list()
                        ),
                    })
                cuda_report["devices"] = devices
            report["torch_cuda"] = cuda_report
        except Exception as exc:
            report["torch_cuda"] = {"installed": True, "error": repr(exc)}

    if report["packages"]["numba"] is not None:
        try:
            numba = importlib.import_module("numba")
            cuda = importlib.import_module("numba.cuda")
            report["numba_cuda"] = {
                "installed": True,
                "numba_version": numba.__version__,
                "available": bool(cuda.is_available()),
                "detected_devices": str(cuda.gpus) if cuda.is_available() else [],
            }
        except Exception as exc:
            report["numba_cuda"] = {"installed": True, "error": repr(exc)}

    torch_devices = report["torch_cuda"].get("devices", [])
    if torch_devices:
        for device in torch_devices:
            if not device["architecture_supported_by_build"]:
                report["assessment"].append(
                    f"PyTorch build does not list {device['name']} "
                    f"(sm_{device['compute_capability'].replace('.', '')}) in its compiled architectures."
                )
    elif not report["torch_cuda"].get("available", False):
        report["assessment"].append(
            "PyTorch CUDA is unavailable; do not assume a CUDA backend can run."
        )

    if not report["numba_cuda"].get("available", False):
        report["assessment"].append(
            "Numba CUDA is unavailable or not installed."
        )

    report["assessment"].append(
        "This is a capability probe only. It does not benchmark kernels or establish CPU/GPU numerical parity."
    )
    output = Path("results/acceleration-capability.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    print(f"\nSaved: {output.resolve()}")


if __name__ == "__main__":
    main()
