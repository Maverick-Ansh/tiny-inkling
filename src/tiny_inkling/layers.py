"""
Primitive layers for Inkling-mini: RMSNorm, SwiGLU MLP, and the depthwise causal
"short convolution" that Inkling sprinkles in two places.

Nothing here is exotic on its own — the interesting part is *where* the short
convs get inserted (see attention.py / model.py). We keep the math explicit and
annotated because the whole point of this repo is to *understand* the recipe.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    r"""Root-mean-square layer norm (no mean subtraction, no bias).

        y = x / sqrt(mean(x^2) + eps) * g

    Cheaper than LayerNorm and the de-facto standard in modern LLMs. We compute
    the normaliser in fp32 for numerical safety even when the model runs in fp16
    (important on T4, which has no bf16).
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.type_as(self.weight) * self.weight).to(dtype)


class SwiGLU(nn.Module):
    r"""Gated SiLU MLP (the standard "SwiGLU" feed-forward block).

        SwiGLU(x) = (SiLU(x W_gate) ⊙ (x W_up)) W_down

    Used both as the *dense* MLP in early layers and as the body of *each expert*
    in the MoE layers. `inter_dim` is the (small) inner width.
    """

    def __init__(self, dim: int, inter_dim: int, bias: bool = False):
        super().__init__()
        self.w_gate = nn.Linear(dim, inter_dim, bias=bias)
        self.w_up = nn.Linear(dim, inter_dim, bias=bias)
        self.w_down = nn.Linear(inter_dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class ShortConv(nn.Module):
    r"""Depthwise **causal** 1-D convolution — Inkling's "short convolution".

    Applied at two points in every attention layer:
      (1) to the K and V streams right after their projections, and
      (2) to the attention- and MLP-branch outputs before they rejoin the
          residual stream.

    Why depthwise + causal?
      * Depthwise (groups = channels) => one tiny kernel per channel, negligible
        params/FLOPs. It mixes information *across time* within a channel, which
        gives each token a cheap, learned, short-range "memory" that complements
        attention. Intuitively it lets the model build local n-gram / smoothing
        features without spending attention capacity on them.
      * Causal => token t only sees t-(k-1) .. t. We enforce this with left
        padding of size (kernel-1) and no right padding, so the layer never
        leaks future information (essential for a decoder LM).

    Shape: x is (B, T, C). We conv along T with a depthwise kernel of width k.
    """

    def __init__(self, channels: int, kernel_size: int = 4):
        super().__init__()
        self.kernel_size = kernel_size
        # groups=channels makes it depthwise. bias adds a per-channel offset.
        self.conv = nn.Conv1d(
            channels, channels, kernel_size,
            groups=channels, bias=True, padding=0,
        )
        # Initialise close to identity-ish: last tap ~1, others ~0, so the conv
        # starts as (almost) a pass-through and learns to deviate. This keeps
        # early training stable — the branch outputs are barely perturbed at t=0.
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[:, :, -1] = 1.0
            self.conv.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, C) -> (B, C, T) for conv1d
        b, t, c = x.shape
        x = x.transpose(1, 2)
        # left-pad by (k-1) so output length == input length and it stays causal
        x = F.pad(x, (self.kernel_size - 1, 0))
        x = self.conv(x)
        return x.transpose(1, 2)  # back to (B, T, C)
