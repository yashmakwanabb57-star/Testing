import pytest

from rosette.perception import device
from rosette.perception.settings import load


def test_defaults():
    s = load({})
    assert (s.device, s.model) == ("auto", "small") and s.models_dir.name == "models"


def test_invalid_value_names_the_variable():
    with pytest.raises(ValueError, match="ROSETTE_SAM2"):
        load({"ROSETTE_SAM2": "huge"})


def test_invalid_device_names_the_variable():
    with pytest.raises(ValueError, match="ROSETTE_DEVICE"):
        load({"ROSETTE_DEVICE": "tpu"})


def test_env_values_are_read(tmp_path):
    s = load({"ROSETTE_DEVICE": "cpu", "ROSETTE_SAM2": "tiny", "ROSETTE_MODELS_DIR": str(tmp_path)})
    assert (s.device, s.model, s.models_dir) == ("cpu", "tiny", tmp_path)


def test_cpu_setting_is_cpu():
    assert device.pick("cpu") == ("cpu", "forced by setting")


def test_cuda_setting_without_cuda_raises(monkeypatch):
    monkeypatch.setattr(device.torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="cuda"):
        device.pick("cuda")


def test_auto_without_cuda_is_cpu(monkeypatch):
    monkeypatch.setattr(device.torch.cuda, "is_available", lambda: False)
    assert device.pick("auto")[0] == "cpu"


def test_auto_with_unsupported_capability_is_cpu(monkeypatch):
    monkeypatch.setattr(device.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(device.torch.cuda, "get_device_capability", lambda: (3, 5))
    monkeypatch.setattr(device.torch.cuda, "get_arch_list", lambda: ["sm_75", "sm_80"])
    d, reason = device.pick("auto")
    assert d == "cpu" and "3.5" in reason


def test_auto_with_supported_capability_is_cuda(monkeypatch):
    monkeypatch.setattr(device.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(device.torch.cuda, "get_device_capability", lambda: (8, 0))
    monkeypatch.setattr(device.torch.cuda, "get_arch_list", lambda: ["sm_75", "sm_80"])
    assert device.pick("auto")[0] == "cuda"


def test_reader_settings_default_to_auto_8089_14():
    s = load({})
    assert (s.reader, s.reader_port, s.reader_threads) == ("auto", 8089, 14)


def test_reader_settings_reject_an_unknown_mode():
    with pytest.raises(ValueError):
        load({"ROSETTE_READER": "maybe"})


def test_reader_settings_are_read_and_bad_numbers_name_the_variable():
    s = load({"ROSETTE_READER": "on", "ROSETTE_READER_PORT": "9000", "ROSETTE_READER_THREADS": "6"})
    assert (s.reader, s.reader_port, s.reader_threads) == ("on", 9000, 6)
    with pytest.raises(ValueError, match="ROSETTE_READER_PORT"):
        load({"ROSETTE_READER_PORT": "eighty"})
