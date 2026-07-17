"""
Attention for Inkling-mini.

Three departures from the vanilla recipe, all annotated inline:

  1. Sliding-window vs global interleave (5:1). Most layers only attend within a
     local window (cheap, O(T·W)); one-in-six is global (O(T²)). This is a KV-cache
     and compute win at long context with little quality loss.

  2. Grouped-query attention (GQA). n_kv_heads < n_heads, so several query heads
     share one K/V head. Shrinks the KV cache by n_heads / n_kv_heads.

  3. Shaw et al. (2018) *relative* position embeddings instead of RoPE. We add a
     learned, distance-clipped bias to the attention logits:

         logit_{i,j} = (q_i · k_j) / √d  +  (q_i · r_{clip(j-i)}) / √d

     where r is a learned table indexed by the *relative* offset j−i, clipped to
     ±k. Clipping is what gives length extrapolation: any offset beyond k reuses
     the boundary embedding, so a model trained at 512 still behaves sensibly at
     2048. (This is the "relative key" term of Shaw; Music Transformer showed how
     to compute it efficiently by re-indexing a single q·rᵀ product.)

  Plus Inkling's short conv on K and V (see ShortConv in layers.py): a depthwise
  causal conv applied to the K and V streams right after projection, giving each
  key/value a cheap local-in-time smoothing before attention.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import InklingConfig
from .layers import ShortConv


class InklingAttention(nn.Module):
    def __init__(self, cfg: InklingConfig, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.is_global = cfg.is_global_layer(layer_idx)
        self.window = cfg.sliding_window

        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.n_rep = cfg.n_heads // cfg.n_kv_heads   # how many Q heads per KV head (GQA)
        self.head_dim = cfg.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Projections. Q is full width; K/V are narrowed to n_kv_heads (GQA).
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)

        # (1) Short conv on K and V — one per KV head-group, depthwise over the
        #     (n_kv_heads*head_dim) channels. Applied after projection, before
        #     splitting into heads.
        self.k_conv = ShortConv(cfg.n_kv_heads * cfg.head_dim, cfg.short_conv_kernel)
        self.v_conv = ShortConv(cfg.n_kv_heads * cfg.head_dim, cfg.short_conv_kernel)

        # (3) Shaw relative-position table: 2k+1 learned vectors of size head_dim,
        #     shared across heads. Index 0 -> offset -k, index 2k -> offset +k.
        self.k_clip = cfg.rel_pos_max_distance
        self.rel_emb = nn.Embedding(2 * self.k_clip + 1, cfg.head_dim)
        nn.init.normal_(self.rel_emb.weight, std=0.02)

    # ---- relative-index buffer, lazily sized to the sequence length ----
    def _rel_index(self, T: int, device) -> torch.Tensor:
        """(T, T) long tensor: entry [i, j] = clip(j - i, -k, +k) + k."""
        pos = torch.arange(T, device=device)
        rel = pos[None, :] - pos[:, None]                 # j - i, shape (T, T)
        rel = rel.clamp(-self.k_clip, self.k_clip) + self.k_clip
        return rel

    def _bias_mask(self, T: int, device, dtype) -> torch.Tensor:
        """Additive attention mask (T, T): 0 where allowed, -inf where masked.

        Always causal (j <= i). Windowed layers additionally forbid j < i-W+1."""
        i = torch.arange(T, device=device)[:, None]
        j = torch.arange(T, device=device)[None, :]
        allowed = j <= i                                   # causal
        if not self.is_global:
            allowed &= (i - j) < self.window               # sliding window
        mask = torch.zeros(T, T, device=device, dtype=dtype)
        mask.masked_fill_(~allowed, float("-inf"))
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, Hk, dh = self.n_heads, self.n_kv_heads, self.head_dim

        q = self.q_proj(x)                                 # (B, T, H*dh)
        k = self.k_proj(x)                                 # (B, T, Hk*dh)
        v = self.v_proj(x)

        # (1) short conv on K and V streams (still in "flat channel" layout)
        k = self.k_conv(k)
        v = self.v_conv(v)

        # reshape to heads: (B, H, T, dh)
        q = q.view(B, T, H, dh).transpose(1, 2)
        k = k.view(B, T, Hk, dh).transpose(1, 2)
        v = v.view(B, T, Hk, dh).transpose(1, 2)

        # GQA: repeat each KV head n_rep times so shapes line up with Q heads.
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)     # (B, H, T, dh)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # ---- content logits: q·kᵀ scaled ----
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale   # (B, H, T, T)

        # ---- (3) Shaw relative-position logits ----
        # q·rᵀ over the 2k+1 relative embeddings, then gather by clipped offset.
        rel_full = torch.einsum("bhid,rd->bhir", q, self.rel_emb.weight)  # (B,H,T,2k+1)
        idx = self._rel_index(T, x.device)                               # (T, T)
        idx = idx.view(1, 1, T, T).expand(B, H, T, T)
        rel_bias = rel_full.gather(-1, idx) * self.scale                 # (B, H, T, T)
        scores = scores + rel_bias

        # ---- causal / window mask ----
        scores = scores + self._bias_mask(T, x.device, scores.dtype)

        # ---- softmax in fp32 for stability (T4 fp16), then weighted sum ----
        attn = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        out = torch.matmul(attn, v)                        # (B, H, T, dh)

        out = out.transpose(1, 2).contiguous().view(B, T, H * dh)
        return self.o_proj(out)
