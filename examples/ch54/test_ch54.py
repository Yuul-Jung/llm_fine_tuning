# 제54장 검증 테스트
#
# 실행: python -m pytest examples/ch54 -q

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pytest  # noqa: E402

pytest.importorskip("gguf")

from gguf import TokenType  # noqa: E402
from safetensors import safe_open  # noqa: E402

from to_gguf import convert, gguf_name, read_summary, vocab  # noqa: E402

TINY = HERE.parents[1] / "models" / "qwen2-tiny-pretrained"
need_tiny = pytest.mark.skipif(not TINY.exists(), reason="제25장의 Qwen2-Tiny 필요")


@need_tiny
def test_every_tensor_has_unique_gguf_name():
    with safe_open(TINY / "model.safetensors", "pt") as f:
        names = [gguf_name(k) for k in f.keys()]
    assert len(names) == len(set(names)) and "token_embd.weight" in names and "blk.3.ffn_down.weight" in names


@need_tiny
def test_vocab_padded_to_model_size():
    names, kinds, merges = vocab(TINY, 151936)
    assert len(names) == len(kinds) == 151936 and merges
    assert names[151645] == "<|im_end|>" and kinds[151645] == TokenType.CONTROL
    assert names[-1].startswith("[PAD") and kinds[-1] == TokenType.UNUSED


@need_tiny
def test_convert_and_read_back(tmp_path):
    info = convert(TINY, tmp_path / "t.gguf")
    s = read_summary(tmp_path / "t.gguf")
    assert info["tied"] and s["kv"]["general.architecture"] == "qwen2" and s["kv"]["qwen2.block_count"] == 4
    assert s["n_tensors"] == info["tensors"] and s["has_template"]


def test_recorded_comparison():
    r = json.loads((HERE / "compare.json").read_text())
    assert r["tiny-f32"]["logits_max_diff"] == 0 and r["fpga-f32"]["logits_max_diff"] == 0
    assert all(v["token_ids_same"] and v["greedy_same"] for v in r.values())
    assert r["fpga-f16"]["logits_max_diff"] < 0.1
    assert r["fpga-f32"]["greedy_same_without_config"] is False                # 생성 설정도 옮겨야 같다
