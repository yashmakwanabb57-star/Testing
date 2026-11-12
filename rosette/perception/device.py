"""Pick the torch device. torch is imported lazily (on first use of `pick` or `device.torch`)."""
from __future__ import annotations


def __getattr__(name: str):
    if name == "torch":
        import torch

        globals()["torch"] = torch
        return torch
    raise AttributeError(name)


def pick(setting: str) -> tuple[str, str]:
    if setting == "cpu":
        return "cpu", "forced by setting"
    if setting not in ("auto", "cuda"):
        raise ValueError(f"unknown device setting {setting!r}; allowed values: auto, cpu, cuda")
    torch = __getattr__("torch")
    available = torch.cuda.is_available()
    if setting == "cuda":
        if not available:
            raise ValueError("ROSETTE_DEVICE=cuda but torch.cuda.is_available() is False")
        return "cuda", "forced by setting"
    if not available:
        return "cpu", "no CUDA device usable (no device, or driver too old for this torch build)"
    major, minor = torch.cuda.get_device_capability()
    # a build with sm_MN runs a device of capability M.x for x >= N (binary compatibility
    # within a major version); anything older than every compiled arch of its major is out
    archs = [(int(a[3]), int(a[4:])) for a in torch.cuda.get_arch_list() if a.startswith("sm_") and a[3:].isdigit()]
    if not any(am == major and an <= minor for am, an in archs):
        return "cpu", f"compute capability {major}.{minor} not in this torch build"
    return "cuda", f"compute capability {major}.{minor}"
