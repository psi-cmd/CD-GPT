"""
Utility modules to upgrade a **CD‑GPT** style Transformer with
(1) **rank‑stabilised LoRA (rsLoRA)** and
(2) **gated Adapters** (UniPELT/AdaMix‑style).

Drop‑in usage:
>>> from hybrid_peft_modules import apply_hybrid_peft
>>> model = CDGPT(...)                 # your frozen backbone
>>> apply_hybrid_peft(model, rank=16)  # inject rsLoRA + Adapters
>>> optimizer = torch.optim.AdamW(get_peft_param_groups(model))

© 2025
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# 引入 CD‑GPT 原始的 RoPE 实现
from model.layer import apply_rotary_emb  # type: ignore


# ------------------------------------------------------------
# 1.  Rank‑Stabilised LoRA Self‑Attention
# ------------------------------------------------------------

class rsLoRASelfAttention(nn.Module):
    """Self‑attention wrapper that adds *rank‑stabilised* LoRA (rsLoRA)
    with a learnable gate per Q/K/V branch.

    Args:
        base_attn: The **frozen** attention module to be wrapped.
        rank:  LoRA rank *r*.
        alpha: LoRA alpha (scaling hyper‑parameter).
    """

    def __init__(self, base_attn: nn.Module, rank: int = 8, alpha: int = 32):
        super().__init__()
        self.base = base_attn  # frozen reference
        self.rank = rank
        self.scale = alpha / math.sqrt(rank)  # rsLoRA scaling α/√r

        # 获取维度
        self.dim = self.base.dim  # 完整隐藏维度
        self.num_heads = self.base.num_heads
        self.head_dim = self.base.head_dim

        # ---- 创建LoRA参数矩阵 ---- #
        # 注意：使用完整维度以匹配整个投影
        self.Aq = nn.Parameter(torch.empty(rank, self.dim))
        self.Bq = nn.Parameter(torch.empty(self.dim, rank))
        self.Ak = nn.Parameter(torch.empty(rank, self.dim))
        self.Bk = nn.Parameter(torch.empty(self.dim, rank))
        self.Av = nn.Parameter(torch.empty(rank, self.dim))
        self.Bv = nn.Parameter(torch.empty(self.dim, rank))

        # 初始化
        nn.init.kaiming_uniform_(self.Aq, a=math.sqrt(5))
        nn.init.zeros_(self.Bq)
        nn.init.kaiming_uniform_(self.Ak, a=math.sqrt(5))
        nn.init.zeros_(self.Bk)
        nn.init.kaiming_uniform_(self.Av, a=math.sqrt(5))
        nn.init.zeros_(self.Bv)

        # 可学习门控 (tanh激活)
        self.gq = nn.Parameter(torch.zeros(()))
        self.gk = nn.Parameter(torch.zeros(()))
        self.gv = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        x: torch.Tensor,           # [B, T, C]
        rope: torch.Tensor,        # [T, 1, hd//2, 2] 
        attn_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        B, T, C = x.shape
        
        # 1) 原始线性投影 (不包含RoPE)
        qkv_raw = self.base.c_attn(x)  # [B, T, 3C]
        
        # 2) LoRA增量计算 (批量方式)
        # 计算所有LoRA增量并连接
        lora_qkv = torch.cat([
            (x @ self.Bq @ self.Aq),  # [B, T, C]
            (x @ self.Bk @ self.Ak),
            (x @ self.Bv @ self.Av)
        ], dim=2) * self.scale
        
        # 应用门控
        gates = torch.cat([
            torch.tanh(self.gq).expand(1, 1, C), 
            torch.tanh(self.gk).expand(1, 1, C),
            torch.tanh(self.gv).expand(1, 1, C)
        ], dim=2)
        
        # 合并原始投影和门控后的LoRA增量
        qkv_adapted = qkv_raw + lora_qkv * gates
        
        # 3) 分割qkv并应用处理
        q_raw, k_raw, v_raw = qkv_adapted.split(C, dim=2)
        
        # 重塑并应用RoPE
        q = q_raw.view(B, T, self.num_heads, self.head_dim)
        k = k_raw.view(B, T, self.num_heads, self.head_dim)
        v = v_raw.view(B, T, self.num_heads, self.head_dim)
        
        # 应用旋转位置编码
        q = apply_rotary_emb(q, rope).transpose(1, 2)  # [B, nh, T, hd]
        k = apply_rotary_emb(k, rope).transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 4) 注意力计算
        y, scores = self.base._scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, need_attn=False
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.base.c_proj(y)
        
        return y, None, scores  # 保持与CD-GPT Block接口一致


# ------------------------------------------------------------
# 2.  Gated Adapter (Houlsby‑style + gate & LN)
# ------------------------------------------------------------

class GatedAdapter(nn.Module):
    """A bottleneck Adapter with a learnable gate (∘ tanh) in the residual path."""

    def __init__(self, dim: int, reduction_factor: int = 8):
        super().__init__()
        hidden = dim // reduction_factor
        self.down = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.up = nn.Linear(hidden, dim)
        self.gate = nn.Parameter(torch.zeros(()))  # init closed
        self.ln = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.up(self.act(self.down(x)))
        x = residual + torch.tanh(self.gate) * y
        return self.ln(x)


# ------------------------------------------------------------
# 3.  Convenience hooks
# ------------------------------------------------------------

def apply_hybrid_peft(model: nn.Module, rank: int = 8, alpha: int = 32):
    """Inject rsLoRA + adapters; ensure adapters move with input device."""
    # 1) freeze backbone
    for p in model.parameters():
        p.requires_grad = False
    # 2) inject
    adapters: List[nn.Module] = []
    for blk in model.transformer.h:  # type: ignore
        # replace attn
        blk.attn = rsLoRASelfAttention(blk.attn, rank=rank, alpha=alpha)
        # create adapter
        adapter = GatedAdapter(model.embedding_dim)
        adapters.append(adapter)
        # register hook
        def make_hook(adp: nn.Module):
            def hook(module, inputs, outputs):
                x_out = outputs[0]
                # move adapter if needed
                if next(adp.parameters()).device != x_out.device:
                    adp.to(x_out.device)
                return (adp(x_out), *outputs[1:])
            return hook
        blk.register_forward_hook(make_hook(adapter))
    model._peft_adapters = nn.ModuleList(adapters)  # type: ignore


def get_peft_param_groups(model: nn.Module, lr_lora: float = 1e-4, lr_adapter: float = 2e-5):
    """Return two‑group param list for AdamW: LoRA vs Adapter."""
    lora_params, adapter_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(tag in n for tag in ('.Aq', '.Ak', '.Av', '.Bq', '.Bk', '.Bv')):
            lora_params.append(p)
        else:
            adapter_params.append(p)
    return [
        {"params": lora_params, "lr": lr_lora, "weight_decay": 0.0},
        {"params": adapter_params, "lr": lr_adapter, "weight_decay": 0.01},
    ]


__all__ = [
    "rsLoRASelfAttention",
    "GatedAdapter",
    "apply_hybrid_peft",
    "get_peft_param_groups",
]
