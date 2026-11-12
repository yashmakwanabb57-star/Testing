import pytest

from rosette.perception.sam2 import MissingModel, load, local_dir
from rosette.perception.settings import DEFAULT_MODELS_DIR, Settings

MODELS = DEFAULT_MODELS_DIR


def test_missing_model_names_the_fetch_command(tmp_path):
    with pytest.raises(MissingModel, match=r"tools/fetch_models.py --model small"):
        load(Settings(device="cpu", model="small", models_dir=tmp_path))


def test_off_is_not_loadable(tmp_path):
    with pytest.raises(ValueError, match="ROSETTE_SAM2"):
        load(Settings(device="cpu", model="off", models_dir=tmp_path))


@pytest.mark.slow
def test_loads_small_on_cpu():
    s = Settings(device="cpu", model="small", models_dir=MODELS)
    if not local_dir(MODELS, "small").joinpath("model.safetensors").exists():
        pytest.skip("weights not fetched: run tools/fetch_models.py --model small")
    ld = load(s)
    assert ld.device == "cpu" and ld.model_id == "facebook/sam2.1-hiera-small"
