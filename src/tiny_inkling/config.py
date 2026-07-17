"""
Configuration for Inkling-mini — a small, faithful re-implementation of the
Inkling / DeepSeek-V3-style Mixture-of-Experts transformer.

We deliberately keep the *mechanisms* identical to the full recipe and only
shrink the *dimensions* so the whole thing trains end-to-end on 2×T4 (16 GB
each) in a few hours. Every field below maps to a concrete design decision that
is annotated in the module where it is consumed.

Scaling note (full Inkling -> ours):
    routed experts   256 -> 32
    shared experts     2 ->  2   (kept identical — shared experts are cheap)
    active routed      6 ->  6   (kept identical — this is "the number")
    SWA:global ratio 5:1 -> 5:1  (kept identical: n_layers=6 => exactly 1 global)
    hidden size     large -> 384
    KV heads (GQA)     8 ->  2
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import json


@dataclass
class InklingConfig:
    # ---- tokenizer / sequence ----
    vocab_size: int = 8192
    max_seq_len: int = 512          # training context; RelPos extrapolates beyond this

    # ---- transformer trunk ----
    d_model: int = 384
    n_layers: int = 6               # 6 layers with global_every=6 => exactly one global layer (5:1)
    n_heads: int = 6                # query heads
    n_kv_heads: int = 2             # GQA: 2 KV heads shared across the 6 query heads (3:1 grouping)
    head_dim: int = 64              # d_model / n_heads = 384/6 = 64

    # ---- attention: sliding-window vs global interleave ----
    # Inkling interleaves sliding-window and global layers at 5:1. We realise this by
    # marking every `global_every`-th layer as global; all others are windowed.
    global_every: int = 6           # layer i is global iff (i % global_every) == (global_every-1)
    sliding_window: int = 256       # local attention span for windowed layers

    # ---- Shaw et al. (2018) relative position embeddings ----
    # We replace RoPE with learned *relative* position biases, clipped to a max
    # distance. This is cheaper for windowed layers (small clip) and, per the
    # Inkling report, extrapolates better to longer sequences than RoPE.
    rel_pos_max_distance: int = 128  # distances are clipped to [-k, +k]; k=128
    rel_pos_num_buckets: int = 0     # 0 => one learned bias per integer distance (Shaw-exact);
                                     # >0 => T5-style log-bucketed (cheaper for global layers)

    # ---- short convolutions (Inkling "conv at two points") ----
    # 1) Depthwise causal conv on K and V *after* their projections.
    # 2) Depthwise causal conv on the attention- and MLP-residual *branch outputs*
    #    before they rejoin the main residual stream.
    short_conv_kernel: int = 4       # small causal kernel; token t sees t-3..t

    # ---- Mixture of Experts ----
    n_routed_experts: int = 32
    n_shared_experts: int = 2
    n_active_routed: int = 6         # top-k routed experts per token
    expert_inter_dim: int = 256      # SwiGLU inner dim of *each* expert (small)
    # aux-loss-free load balancing (DeepSeek-V3): a per-expert bias is added to the
    # routing score ONLY for top-k selection (not for the weighting), and is nudged
    # up/down by a fixed step to equalise expert load. No gradient, no aux loss.
    router_bias_update_speed: float = 1e-3
    # Optional tiny "sequence-wise" aux loss kept at ~0; DeepSeek-V3 keeps a small
    # complementary term. We expose it but default it off to stay aux-loss-*free*.
    aux_loss_alpha: float = 0.0
    router_score_func: str = "sigmoid"   # sigmoid gates (not softmax) — scores are independent

    # ---- normalisation / misc ----
    rms_norm_eps: float = 1e-5
    tie_embeddings: bool = True
    dropout: float = 0.0

    # ---- MoE placement ----
    # First `n_dense_layers` use a plain SwiGLU MLP (DeepSeek warms up with dense
    # layers before going sparse — stabilises early routing). Rest are MoE.
    n_dense_layers: int = 1

    def is_global_layer(self, i: int) -> bool:
        return (i % self.global_every) == (self.global_every - 1)

    def is_moe_layer(self, i: int) -> bool:
        return i >= self.n_dense_layers

    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str):
        with open(path) as f:
            return cls(**json.load(f))


# A couple of named presets used by the scripts.
TINY = InklingConfig()  # ~ the numbers above; ~70M params, ~20M active

DEBUG = InklingConfig(
    vocab_size=8192, max_seq_len=128, d_model=128, n_layers=4, n_heads=4,
    n_kv_heads=2, head_dim=32, n_routed_experts=8, n_active_routed=2,
    expert_inter_dim=128, sliding_window=64, rel_pos_max_distance=32,
    global_every=4, n_dense_layers=1,
)
