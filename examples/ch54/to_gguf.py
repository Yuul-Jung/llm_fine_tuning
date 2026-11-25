# 제54장: Hugging Face(Safetensors) Qwen2 모델을 GGUF 한 파일로 바꾸는 변환기
#
# GGUF 쓰기·읽기는 llama.cpp 프로젝트의 파이썬 라이브러리(gguf)를 쓴다.
# 텐서 이름과 메타데이터 열쇠는 그 라이브러리가 정의한 Qwen2 형식을 따른다.

import json
from pathlib import Path

import numpy as np
import torch
from gguf import GGMLQuantizationType, GGUFWriter, LlamaFileType, TokenType, quants
from safetensors.torch import load_file

NAME_MAP = {                                                 # Hugging Face 이름 → GGUF 이름
    "model.embed_tokens.weight": "token_embd.weight",
    "model.norm.weight": "output_norm.weight",
    "lm_head.weight": "output.weight",
}
LAYER_MAP = {
    "input_layernorm.weight": "attn_norm.weight",
    "self_attn.q_proj.weight": "attn_q.weight", "self_attn.q_proj.bias": "attn_q.bias",
    "self_attn.k_proj.weight": "attn_k.weight", "self_attn.k_proj.bias": "attn_k.bias",
    "self_attn.v_proj.weight": "attn_v.weight", "self_attn.v_proj.bias": "attn_v.bias",
    "self_attn.o_proj.weight": "attn_output.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight", "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
}
FILE_TYPES = {"F32": (LlamaFileType.ALL_F32, np.float32), "F16": (LlamaFileType.MOSTLY_F16, np.float16),
              "Q8_0": (LlamaFileType.MOSTLY_Q8_0, None), "Q4_0": (LlamaFileType.MOSTLY_Q4_0, None)}   # 제55장


def gguf_name(hf: str) -> str:
    if hf in NAME_MAP:
        return NAME_MAP[hf]
    _, _, layer, rest = hf.split(".", 3)                     # model.layers.{i}.{나머지}
    return f"blk.{layer}.{LAYER_MAP[rest]}"


def vocab(model_dir: Path, vocab_size: int) -> tuple:
    """토큰 문자열·종류·병합 규칙. 모델의 어휘 크기까지 빈 칸은 PAD로 채운다(llama.cpp 변환기와 같은 방식)."""
    tj = json.loads((model_dir / "tokenizer.json").read_text(encoding="utf-8"))
    tokens = {i: t for t, i in tj["model"]["vocab"].items()}
    types = {i: TokenType.NORMAL for i in tokens}
    for a in tj["added_tokens"]:
        tokens[a["id"]] = a["content"]
        types[a["id"]] = TokenType.CONTROL if a["special"] else TokenType.USER_DEFINED
    names = [tokens.get(i, f"[PAD{i}]") for i in range(vocab_size)]
    kinds = [int(types.get(i, TokenType.UNUSED)) for i in range(vocab_size)]
    merges = [m if isinstance(m, str) else " ".join(m) for m in tj["model"]["merges"]]
    return names, kinds, merges


def convert(model_dir: Path, out: Path, ftype: str = "F32") -> dict:
    cfg = json.loads((model_dir / "config.json").read_text())
    tok_cfg = json.loads((model_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    file_type, np_dtype = FILE_TYPES[ftype]
    w = GGUFWriter(str(out), "qwen2")
    w.add_name(model_dir.name)
    w.add_context_length(cfg["max_position_embeddings"])
    w.add_embedding_length(cfg["hidden_size"])
    w.add_block_count(cfg["num_hidden_layers"])
    w.add_feed_forward_length(cfg["intermediate_size"])
    w.add_head_count(cfg["num_attention_heads"])
    w.add_head_count_kv(cfg["num_key_value_heads"])
    w.add_rope_freq_base(float(cfg.get("rope_theta") or cfg.get("rope_parameters", {}).get("rope_theta", 1e6)))
    w.add_layer_norm_rms_eps(cfg["rms_norm_eps"])
    w.add_file_type(int(file_type))
    names, kinds, merges = vocab(model_dir, cfg["vocab_size"])
    w.add_tokenizer_model("gpt2")                            # 바이트 단위 BPE
    w.add_tokenizer_pre("qwen2")
    w.add_token_list(names)
    w.add_token_types(kinds)
    w.add_token_merges(merges)
    w.add_eos_token_id(cfg["eos_token_id"] if isinstance(cfg["eos_token_id"], int) else cfg["eos_token_id"][0])
    w.add_bos_token_id(cfg.get("bos_token_id") or 151643)  # 설정에 없으면 endoftext(Qwen2.5와 같음)
    w.add_pad_token_id(151643)                               # endoftext(제45장)
    w.add_add_bos_token(False)
    template = model_dir / "chat_template.jinja"
    w.add_chat_template(template.read_text() if template.exists() else tok_cfg["chat_template"])

    state = {}
    for f in sorted(model_dir.glob("*.safetensors")):
        state.update(load_file(f))
    counts = {"F32": 0, ftype: 0}
    for hf, t in state.items():
        a = t.to(torch.float32).numpy()
        if a.ndim == 2 and ftype.startswith("Q"):            # 제55장: 32개 블록 양자화(gguf 라이브러리 구현)
            q = GGMLQuantizationType[ftype]
            w.add_tensor(gguf_name(hf), quants.quantize(a, q), raw_dtype=q)
            counts[ftype] += 1
            continue
        if a.ndim == 2 and ftype != "F32":                   # 행렬만 낮은 정밀도, 1차원(편향·정규화)은 F32
            a = a.astype(np_dtype)
            counts[ftype] += 1
        else:
            counts["F32"] += 1
        w.add_tensor(gguf_name(hf), a)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return {"tensors": len(state), "tied": "lm_head.weight" not in state, "types": counts}


def read_summary(path: Path) -> dict:
    from gguf import GGUFReader
    r = GGUFReader(str(path))
    kv = {}
    for k in ("general.architecture", "qwen2.block_count", "qwen2.attention.head_count_kv",
              "tokenizer.ggml.model", "tokenizer.ggml.eos_token_id"):
        field = r.fields[k]
        v = field.parts[field.data[0]]
        kv[k] = bytes(v).decode() if v.dtype == np.uint8 else v.tolist()[0]
    t0 = r.tensors[0]
    return {"kv": kv, "n_tensors": len(r.tensors), "n_fields": len(r.fields),
            "first": (t0.name, [int(x) for x in t0.shape], GGMLQuantizationType(t0.tensor_type).name),
            "has_template": "tokenizer.chat_template" in r.fields}
