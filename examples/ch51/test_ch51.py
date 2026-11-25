# 제51장 검증 테스트
#
# 실행: python -m pytest examples/ch51 -q

import json
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pytest  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

from save_load import CKPT, FINAL, MANIFEST, QWEN_REVISION, read_header, sha256, tensor_by_hand  # noqa: E402


def test_safetensors_layout_by_hand(tmp_path):
    t = {"a": torch.arange(6, dtype=torch.float32).reshape(2, 3), "b": torch.ones(4)}
    save_file(t, tmp_path / "x.safetensors")
    n, header, body = read_header(tmp_path / "x.safetensors")
    assert struct.unpack("<Q", (tmp_path / "x.safetensors").read_bytes()[:8])[0] == n
    assert header["a"]["shape"] == [2, 3] and len(body) == (6 + 4) * 4
    assert all(torch.equal(tensor_by_hand(header, body, k), t[k]) for k in t)


@pytest.mark.skipif(not FINAL.exists(), reason="실습을 먼저 실행해야 함")
def test_saved_adapter_matches_checkpoint_and_records_revision():
    w1, w2 = load_file(CKPT / "adapter_model.safetensors"), load_file(FINAL / "adapter_model.safetensors")
    assert w1.keys() == w2.keys() and all(torch.equal(w1[k], w2[k]) for k in w1)
    cfg = json.loads((FINAL / "adapter_config.json").read_text())
    assert cfg["revision"] == QWEN_REVISION and cfg["r"] == 8 and cfg["lora_alpha"] == 16


@pytest.mark.skipif(not FINAL.exists(), reason="실습을 먼저 실행해야 함")
def test_manifest_hash_matches_file():
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert m["base_revision"] == QWEN_REVISION
    assert m["adapter_sha256"] == sha256(FINAL / "adapter_model.safetensors")


def test_adapter_shapes_match_qwen():
    _, header, _ = read_header(CKPT / "adapter_model.safetensors") if CKPT.exists() else pytest.skip("없음")
    shp = {k: v["shape"] for k, v in header.items() if k != "__metadata__"}
    assert len(shp) == 336                                               # 24층 × 7모듈 × (A, B)
    assert shp["base_model.model.model.layers.0.self_attn.k_proj.lora_B.weight"] == [128, 8]
    assert sum(a * b for a, b in shp.values()) == 4_399_104
