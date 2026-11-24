# 제19장 핵심 실습: 직접 만든 Qwen2-Tiny를 저장하고 Hugging Face Qwen2ForCausalLM으로 다시 불러 비교
#
# 실행: python examples/ch19/hf_compat.py
#
# 저장 위치: 저장소 최상위 models/qwen2-tiny/ (.gitignore로 커밋 대상에서 제외)

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402
from transformers.utils import logging  # noqa: E402

from common.qwen import load_tokenizer  # noqa: E402
from tiny_model import TINY, Qwen2Tiny, init_weights  # noqa: E402

OUT = HERE.parents[1] / "models" / "qwen2-tiny"
logging.disable_progress_bar()                              # 진행 막대 숨김
TEXT = "작은 모델도 큰 모델과 같은 길을 걷는다."


def hf_config() -> Qwen2Config:
    """TINY 설계를 Hugging Face 설정 객체로 옮긴다. 특수 토큰 번호는 Qwen2.5 토크나이저와 같게."""
    return Qwen2Config(**TINY, tie_word_embeddings=True, max_position_embeddings=2048,
                       eos_token_id=151645, pad_token_id=151643)


def to_hf(mine: Qwen2Tiny) -> Qwen2ForCausalLM:
    hf = Qwen2ForCausalLM(hf_config())
    missing, unexpected = hf.load_state_dict(mine.state_dict(), strict=True)
    assert not missing and not unexpected                  # 이름·Shape가 하나라도 다르면 여기서 멈춘다
    return hf.eval()


@torch.no_grad()
def greedy(step, ids, n=6):
    for _ in range(n):
        nxt = step(ids)[:, -1].argmax(-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
    return ids[0, -n:].tolist()


def main() -> None:
    tok = load_tokenizer()
    ids = tok(TEXT, return_tensors="pt").input_ids
    mine = init_weights(Qwen2Tiny(), seed=0).eval()

    print("── 1. 같은 가중치를 Qwen2ForCausalLM에 싣기 ──")
    hf = to_hf(mine)
    print(f"  load_state_dict(strict=True) 통과, 파라미터 {hf.num_parameters():,}개")

    print("\n── 2. 저장(save_pretrained) ──")
    hf.save_pretrained(OUT)
    tok.save_pretrained(OUT)                                # 이후 장에서 함께 쓰도록 토크나이저도 저장
    with safe_open(OUT / "model.safetensors", "pt") as f:
        keys = list(f.keys())
    cfg = json.loads((OUT / "config.json").read_text())
    print(f"  model.safetensors 텐서 {len(keys)}개, lm_head 저장 여부: {'lm_head.weight' in keys}")
    print(f"  파일 크기 {(OUT / 'model.safetensors').stat().st_size / 1e6:.1f}MB, "
          f"config: hidden {cfg['hidden_size']}, 계층 {cfg['num_hidden_layers']}, "
          f"tie {cfg['tie_word_embeddings']}")

    print("\n── 3. 다시 불러 순전파 비교 ──")
    loaded = Qwen2ForCausalLM.from_pretrained(OUT, dtype=torch.float32).eval()
    with torch.no_grad():
        a = mine(ids)
        b = loaded(ids).logits
    print(f"  입력 토큰 {ids.shape[1]}개, Logits {tuple(a.shape)}")
    print(f"  최대 오차 {(a - b).abs().max().item():.1e}, "
          f"상대 오차 {((a - b).norm() / b.norm()).item():.1e}")
    print(f"  위치별 1등 토큰 일치: {torch.equal(a.argmax(-1), b.argmax(-1))}")
    same_tie = loaded.lm_head.weight.data_ptr() == loaded.model.embed_tokens.weight.data_ptr()
    print(f"  불러온 모델도 Weight Tying 유지: {same_tie}")

    print("\n── 4. Greedy 생성 6토큰 비교(학습 전이라 뜻은 없다) ──")
    g1 = greedy(mine, ids)
    g2 = greedy(lambda x: loaded(x).logits, ids)
    print(f"  직접 구현 {g1}\n  HF 모델   {g2}\n  일치: {g1 == g2}")


if __name__ == "__main__":
    main()
