# 제29장: 기존 Linear를 감싸 원본은 동결하고 A·B만 학습하는 LoRALinear
#
# y = base(x) + (α/r) · B (A x)            base: 원래 nn.Linear(동결, 편향 포함)

import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base, self.r, self.scale = base, r, alpha / r
        for p in self.base.parameters():
            p.requires_grad_(False)                                   # 원본 동결
        device, dtype = base.weight.device, base.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))         # nn.Linear와 같은 방식의 작은 난수

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * ((x @ self.lora_A.T) @ self.lora_B.T)

    @torch.no_grad()
    def merged(self) -> nn.Linear:
        """W' = W + (α/r)·BA를 담은 보통의 nn.Linear를 새로 만들어 돌려준다(원본은 그대로)."""
        out = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None,
                        device=self.base.weight.device, dtype=self.base.weight.dtype)
        out.weight.copy_(self.base.weight + self.scale * (self.lora_B @ self.lora_A))
        if self.base.bias is not None:
            out.bias.copy_(self.base.bias)
        return out


ALL_LINEAR = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def apply_lora(model: nn.Module, targets=("q_proj", "v_proj"), r: int = 8, alpha: float = 16.0,
               seed: int = 0) -> list:
    """모델 전체를 동결한 뒤, 이름이 targets인 Linear를 LoRALinear로 바꾼다. A의 난수는 seed로 고정한다."""
    torch.manual_seed(seed)
    for p in model.parameters():
        p.requires_grad_(False)
    replaced = []
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if child_name in targets and isinstance(child, nn.Linear):
                setattr(module, child_name, LoRALinear(child, r, alpha))
                replaced.append(f"{name}.{child_name}")
    return replaced


def merge_lora(model: nn.Module) -> int:
    """모든 LoRALinear를 병합한 nn.Linear로 되돌린다. 바꾼 개수를 돌려준다."""
    count = 0
    for module in list(model.modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                setattr(module, child_name, child.merged())
                count += 1
    return count


def trainable_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
