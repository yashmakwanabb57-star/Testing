"""Pick the torch device. torch is imported lazily (on first use of `pick` or `device.torch`)."""
from __future__ import annotations


def __getattr__(name: str):
    if name == "torch":
        import torch

        globals()["torch"] = torch
        return torch
    raise AttributeError(name)


def pick(setting: str) -> tuple[str, str]:
    torch = __getattr__("torch")
    if setting == "cpu":
        return "cpu", "forced by setting"
    available = torch.cuda.is_available()
    if setting == "cuda":
        if not available:
            raise ValueError("ROSETTE_DEVICE=cuda but torch.cuda.is_available() is False")
        return "cuda", "forced by setting"
    if setting != "auto":
        raise ValueError(f"unknown device setting {setting!r}; allowed values: auto, cpu, cuda")
    if not available:
        return "cpu", "no CUDA device usable (no device, or driver too old for this torch build)"
    major, minor = torch.cuda.get_device_capability()
    if f"sm_{major}{minor}" not in torch.cuda.get_arch_list():
        return "cpu", f"compute capability {major}.{minor} not in this torch build"
    return "cuda", f"compute capability {major}.{minor}"
