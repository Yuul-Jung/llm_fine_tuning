# 실제 Qwen2.5-0.5B-Instruct 모델과 토크나이저를 불러오는 공통 함수
#
# 교재 원칙: Revision 고정, FP32로 불러와 CPU에서 계산(GT 1030은 VRAM 2GB라 FP32 전체 모델을 올릴 수 없음)

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from common.models import QWEN_ID, QWEN_REVISION


def load_config(local_only: bool = False):
    return AutoConfig.from_pretrained(QWEN_ID, revision=QWEN_REVISION,
                                      local_files_only=local_only)


def load_tokenizer(local_only: bool = False):
    return AutoTokenizer.from_pretrained(QWEN_ID, revision=QWEN_REVISION,
                                         local_files_only=local_only)


def load_model(local_only: bool = False):
    """FP32로 불러와 평가 모드(eval)로 돌려준다. 약 2GB의 시스템 RAM이 필요하다."""
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_ID, revision=QWEN_REVISION, dtype=torch.float32,
        local_files_only=local_only)
    return model.eval()
