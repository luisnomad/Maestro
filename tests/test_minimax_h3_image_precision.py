import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from models.minimax_h3 import image_handler
from models.minimax_h3.minimax_h3_handler import _text_encoder_variants

DEFAULTS = Path(__file__).resolve().parents[1] / "app" / "defaults"


def _preset(name):
    return json.loads((DEFAULTS / f"{name}.json").read_text())["model"]


@pytest.mark.parametrize(
    "preset, encoder",
    [("h3_image", "nvfp4_awq"), ("h3_image_int8_encoder", "int8"), ("h3_image_bf16", "int8")],
)
def test_presets_select_their_text_encoder(preset, encoder):
    model_def = image_handler.family_handler.query_model_def("h3_image", _preset(preset))
    urls = _text_encoder_variants()[encoder]["URLs"]
    assert model_def["h3_image_text_encoder"] == encoder
    assert model_def["text_encoder_URLs"] == urls
    filename = Path(urls[0]).name
    assert [f"minimax_h3/{filename}", f"Qwen3-VL-32B-Instruct/{filename}"] in model_def["required_model_asset_groups"]


def test_bf16_preset_pins_the_comfy_bf16_transformer():
    model = _preset("h3_image_bf16")
    (filename,) = model["URLs"]
    source = model["download_sources"][filename]
    assert filename == "minimax_h3_image_pruned_bf16.safetensors"
    assert source["size_bytes"] == 40234657162
    assert len(source["sha256"]) == 64
    assert "5e60435c12302e0fa946b35a309736eb25b9272b" in source["url"]


def test_unknown_text_encoder_is_rejected():
    with pytest.raises(ValueError, match="text encoder"):
        image_handler.family_handler.query_model_def("h3_image", {"h3_image_text_encoder": "gguf_q2_k"})
