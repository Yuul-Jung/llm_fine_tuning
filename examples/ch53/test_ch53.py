# 제53장 검증 테스트
#
# 실행: python -m pytest examples/ch53 -q

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from export import MERGED, load_all, save_all  # noqa: E402


def test_four_formats_roundtrip_bit_exact(tmp_path):
    rng = np.random.default_rng(0)
    w = (rng.standard_normal((5, 7)) * 1e-3).astype(np.float32)
    w[0, 0], w[0, 1] = np.float32(1.0), np.float32(-0.0)
    paths = save_all(w, np.zeros(5, np.float32), tmp_path)
    for arr in load_all(paths, w.shape).values():
        assert arr.dtype == np.float32 and arr.view(np.uint32).tolist() == w.view(np.uint32).tolist()
    lines = paths["hex"].read_text().split()
    assert len(lines) == 35 and all(len(x) == 8 for x in lines) and lines[0] == "3f800000"
    assert paths["bin"].stat().st_size == 35 * 4


def test_byte_order():
    b = np.array([1.0], dtype="<f4").tobytes()
    assert b == bytes([0, 0, 0x80, 0x3F]) and np.frombuffer(b, dtype=">f4")[0] != 1.0


def test_linear_reproduction_exact():
    torch.manual_seed(0)
    layer = torch.nn.Linear(16, 4)
    w = np.frombuffer(layer.weight.detach().numpy().astype("<f4").tobytes(), dtype="<f4").reshape(4, 16)
    x = torch.randn(3, 16)
    y = torch.nn.functional.linear(x, torch.from_numpy(w.copy()), layer.bias.detach())
    assert torch.equal(y, layer(x).detach())


@pytest.mark.skipif(not MERGED.exists(), reason="제52장의 병합 모델 필요")
def test_merged_file_stores_tied_weight_once():
    from safetensors import safe_open
    with safe_open(MERGED / "model.safetensors", "pt") as f:
        keys = list(f.keys())
    assert "model.embed_tokens.weight" in keys and "lm_head.weight" not in keys
