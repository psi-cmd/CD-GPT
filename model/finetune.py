from typing import Optional
import torch
import torch.nn as nn

from model.output_head import SequencePredictionHead
from .cd_gpt import CDGPT
from .layer import CasualSelfAttention, apply_rotary_emb
import math

    
class LoRASelfAttention(CasualSelfAttention):
    def __init__(self, dim, num_heads, max_len, rank=8, alpha=32):
        super().__init__(dim, num_heads, max_len)
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Initialize LoRA parameters
        self.lora_q_A = nn.Parameter(torch.zeros(rank, self.dim))
        self.lora_q_B = nn.Parameter(torch.zeros(self.dim, rank))
        self.lora_k_A = nn.Parameter(torch.zeros(rank, self.dim))
        self.lora_k_B = nn.Parameter(torch.zeros(self.dim, rank))
        self.lora_v_A = nn.Parameter(torch.zeros(rank, self.dim))
        self.lora_v_B = nn.Parameter(torch.zeros(self.dim, rank))
        
        # Initialize LoRA parameters
        nn.init.kaiming_uniform_(self.lora_q_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_q_B)
        nn.init.kaiming_uniform_(self.lora_k_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_k_B)
        nn.init.kaiming_uniform_(self.lora_v_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_v_B)

    def _project_qkv(self, x: torch.Tensor, rope: torch.Tensor):
        B, T, C = x.size()
        
        # Original projection
        qkv = self.c_attn(x)
        
        # LoRA adaptation
        lora_qkv = torch.cat([
            (x @ self.lora_q_A.T @ self.lora_q_B.T),
            (x @ self.lora_k_A.T @ self.lora_k_B.T),
            (x @ self.lora_v_A.T @ self.lora_v_B.T)
        ], dim=2) * self.scaling
        
        # Merge
        qkv = qkv + lora_qkv
        q, k, v = qkv.split(self.dim, dim=2)
        
        # Reshape and apply RoPE
        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim)
        v = v.view(B, T, self.num_heads, self.head_dim)
        
        q = apply_rotary_emb(q, rope).transpose(1, 2)
        k = apply_rotary_emb(k, rope).transpose(1, 2)
        v = v.transpose(1, 2)
        
        return q, k, v

class Adapter(nn.Module):
    def __init__(self, dim, reduction_factor=8):
        super().__init__()
        self.down_project = nn.Linear(dim, dim // reduction_factor)
        self.activation = nn.GELU()
        self.up_project = nn.Linear(dim // reduction_factor, dim)
        self.layer_norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        residual = x
        x = self.down_project(x)
        x = self.activation(x)
        x = self.up_project(x)
        x = residual + x
        return self.layer_norm(x)

class CDGPTFineTune(CDGPT):
    def __init__(self, cfg, use_lora=True, use_adapter=True):
        super().__init__(
            vocab_size=cfg.tokenizer.vocab_size,
            max_len=cfg.model.max_len,
            embedding_dim=cfg.model.num_hiddens,
            num_layers=cfg.model.num_layers,
            num_heads=cfg.model.num_heads,
            bias=cfg.model.bias,
            eps=cfg.model.eps,
            include_head=False
        )
        
        self.use_lora = use_lora
        self.use_adapter = use_adapter
        
        if use_lora:
            # Replace attention layer Q,K,V projections with LoRA versions
            for block in self.transformer.h:
                # Replace query, key, value projections
                block.attn = LoRASelfAttention(self.embedding_dim, self.num_heads, self.max_len)
        
        if use_adapter:
            # Add Adapter after each transformer block
            self.adapters = nn.ModuleList([
                Adapter(self.embedding_dim) 
                for _ in range(cfg.model.num_layers)
            ])
        
        # Freeze original model parameters
        for param in self.parameters():
            param.requires_grad = False
            
        # Unfreeze LoRA and Adapter parameters
        if use_lora:
            for block in self.transformer.h:
                block.attn.lora_q_A.requires_grad = True
                block.attn.lora_q_B.requires_grad = True
                block.attn.lora_k_A.requires_grad = True
                block.attn.lora_k_B.requires_grad = True
                block.attn.lora_v_A.requires_grad = True
                block.attn.lora_v_B.requires_grad = True

        if use_adapter:
            for adapter in self.adapters:
                for param in adapter.parameters():
                    param.requires_grad = True

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, pos_ids: Optional[torch.Tensor] = None):
        bs, seq_len = input_ids.shape
        device = input_ids.device
        dtype = input_ids.dtype
        assert (
                seq_len <= self.max_len
        ), f"Cannot forward sequence of length {seq_len}, max length is only {self.max_len}"
        if self.rope_cache is None:
            self.rope_cache = self._make_rope_mask(device, dtype)  # [max_len, ...]

        if pos_ids is not None:
            rope = self.rope_cache.index_select(0, pos_ids)
            if attention_mask is None:
                attention_mask = self._make_casual_mask(device)
            attention_mask = attention_mask.index_select(2, pos_ids)
            attention_mask = attention_mask[:, :, :, :self.max_len]
        else:
            rope = self.rope_cache[:seq_len]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :, :seq_len, :seq_len]

        x = self._forward_embedding_impl(input_ids)
        if pos_ids is None:
            for i, block in enumerate(self.transformer.h):
                if self.activation_checkpoint:
                    x, _, _ = self.activation_checkpoint_func(block, x, rope, attention_mask)
                else:
                    x, _, _ = block(x, rope, attn_mask=attention_mask)
                if self.use_adapter:
                    x = self.adapters[i](x)
        else:
            if not self.kv_caches:
                head_dim = self.embedding_dim // self.num_heads
                cache_shape = (bs, self.num_heads, self.max_len, head_dim)
                # prelocate memory
                self.kv_caches = [
                    (torch.zeros(cache_shape, device=x.device, dtype=x.dtype),
                     torch.zeros(cache_shape, device=x.device, dtype=x.dtype))
                    for _ in range(self.num_layers)
                ]
            for i, block in enumerate(self.transformer.h):
                x, self.kv_caches[i], _ = block(x, rope,
                                                attn_mask=attention_mask,
                                                pos_ids=pos_ids,
                                                kv_cache=self.kv_caches[i])
                if self.use_adapter:
                    x = self.adapters[i](x)
        x = self.transformer.ln_f(x)
        x = self._forward_head_impl(x)
        return x


class CDGPTSequenceTaskHead(nn.Module):
    """
    Output for sequence level task.
    """
    @classmethod
    def from_config(cls, cfg):
        pad_id = cfg.tokenizer.pad_id
        num_classes = cfg.model.num_classes
        return {
            "num_classes": num_classes,
            "pad_id": pad_id,
            "vocab_size": cfg.tokenizer.vocab_size,
            "embedding_dim": cfg.model.num_hiddens
        }

    def __init__(self,
                 embedding_dim=2304,
                 pad_id=2,
                 dropout=0.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.pad_id = pad_id
        self.dropout = dropout
        
        
        # Create sequence prediction head
        self.cls_head = SequencePositiveOutputHead(self.embedding_dim, self.dropout)
        
    def forward(self, input_ids: torch.Tensor,
                hiddens: torch.Tensor):
        """
        Directly process input and return sequence prediction results without calling transformer
        
        Args:
            input_ids: [bs, seq_len], input token indices
        """
        
        # Use last token for classification or regression
        if self.pad_id is None:
            sequence_lengths = -1  
        else:
            sequence_lengths = torch.ne(input_ids, self.pad_id).sum(-1) - 1
        
        batch_size = hiddens.shape[0]
        hiddens = hiddens[torch.arange(batch_size, device=hiddens.device), sequence_lengths]
        
        res = self.cls_head(hiddens)
        return res


class SequencePositiveOutputHead(nn.Module):
    """Head for sequence level classification tasks."""

    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.num_layers = 1
        self.dense = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.out_proj = nn.Sequential(
            nn.Linear(dim, 1),
        )

    def forward(self, features):
        # [B, T, C]
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        # Compress dimensions from [batch_size, 1] to [batch_size]
        x = x.squeeze(-1)
        return x