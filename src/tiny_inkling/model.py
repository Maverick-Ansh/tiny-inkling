"""
The Inkling-mini model: embedding -> N decoder blocks -> RMSNorm -> LM head.

Each block is pre-norm and looks like:

    h = x + shortconv_attn( Attention( norm1(x) ) )
    y = h + shortconv_mlp ( FFN(       norm2(h) ) )

where FFN is a dense SwiGLU for the first `n_dense_layers` and an InklingMoE
afterwards. The two `shortconv_*` are Inkling's *second* short-conv insertion
point: a depthwise causal conv applied to each residual **branch output** before
it is added back to the stream. (The *first* insertion point — on K and V — lives
inside InklingAttention.)

Why conv the branch output? It lets the block apply a cheap, learned, causal
smoothing/gating over time to whatever the attention or FFN just produced, before
committing it to the residual stream. Initialised near-identity (see ShortConv),
so at step 0 the block is a normal pre-norm transformer block and it *learns* to
use the conv.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from .config import InklingConfig
from .layers import RMSNorm, SwiGLU, ShortConv
from .attention import InklingAttention
from .moe import InklingMoE


class InklingBlock(nn.Module):
    def __init__(self, cfg: InklingConfig, layer_idx: int):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.attn = InklingAttention(cfg, layer_idx)
        self.attn_conv = ShortConv(cfg.d_model, cfg.short_conv_kernel)

        self.norm2 = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.is_moe = cfg.is_moe_layer(layer_idx)
        if self.is_moe:
            self.ffn = InklingMoE(cfg)
        else:
            # dense warm-up layer: a wider single SwiGLU (~ n_active experts worth)
            self.ffn = SwiGLU(cfg.d_model, cfg.expert_inter_dim * cfg.n_active_routed)
        self.ffn_conv = ShortConv(cfg.d_model, cfg.short_conv_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # attention branch, conv'd before rejoining
        x = x + self.attn_conv(self.attn(self.norm1(x)))
        # FFN/MoE branch, conv'd before rejoining
        x = x + self.ffn_conv(self.ffn(self.norm2(x)))
        return x


class InklingForCausalLM(nn.Module):
    def __init__(self, cfg: InklingConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([InklingBlock(cfg, i) for i in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        self.apply(self._init)
        # scaled init for residual projections (GPT-2 style: 1/sqrt(2*n_layers))
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("w_down.weight"):
                nn.init.normal_(p, std=0.02 / (2 * cfg.n_layers) ** 0.5)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        x = self.embed(input_ids)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            # standard shifted-CE; ignore_index=-100 lets callers mask tokens (SFT/RL)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)).float(),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return {"logits": logits, "loss": loss}

    # ---- convenience ----
    def num_params(self, only_active: bool = False):
        """Total params, or an estimate of params *active per token*."""
        total = sum(p.numel() for p in self.parameters())
        if not only_active:
            return total
        # active = everything except the routed experts that aren't picked
        cfg = self.cfg
        per_expert = sum(p.numel() for p in self.blocks[-1].ffn.routed_experts[0].parameters()) \
            if cfg.is_moe_layer(cfg.n_layers - 1) else 0
        n_moe_layers = sum(1 for i in range(cfg.n_layers) if cfg.is_moe_layer(i))
        inactive = n_moe_layers * (cfg.n_routed_experts - cfg.n_active_routed) * per_expert
        return total - inactive

    @torch.no_grad()
    def moe_stats(self):
        """Aggregate the latest routing diagnostics across MoE layers."""
        stats = [b.ffn.last_stats for b in self.blocks if getattr(b, "is_moe", False) and b.ffn.last_stats]
        if not stats:
            return {}
        keys = stats[0].keys()
        return {k: sum(s[k] for s in stats) / len(stats) for k in keys}
