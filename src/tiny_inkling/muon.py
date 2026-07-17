"""
Hybrid optimiser: Muon for 2-D weight matrices, AdamW for everything else, with
weight decay coupled to the square of the learning rate.

Muon (Momentum + Orthogonalisation via Newton–Schulz), Keller Jordan et al. 2024
-------------------------------------------------------------------------------
Adam adapts a *per-coordinate* step. Muon instead takes the momentum buffer M
(a matrix) and replaces it with the nearest *semi-orthogonal* matrix before the
step:

    O = NewtonSchulz5(M)         # ≈ U Vᵀ  where  M = U Σ Vᵀ  (SVD)
    W ← W − lr · O

Orthogonalising equalises the spectrum of the update (all singular values → 1),
so no single direction dominates. Empirically this lets the large matmul weights
of a transformer take much larger, better-conditioned steps than Adam. We never
form the SVD; the quintic Newton–Schulz iteration

    X ← a·X + b·(X Xᵀ)X + c·(X Xᵀ)² X

converges to the orthogonal factor from a spectrally-normalised start, using only
matmuls (GPU-friendly, fp16/bf16-safe with the standard (a,b,c) coefficients).

Which params go where
---------------------
  * 2-D weight matrices (attention projections, expert/MLP weights, router) -> Muon.
  * Everything with ndim != 2 (embeddings, RMSNorm gains, conv kernels, biases,
    the relative-position table) -> AdamW. Muon's orthogonalisation is only
    meaningful for matrices, and embeddings/norms behave better under Adam.

Weight decay ∝ lr²  (modular-manifold-inspired coupling)
--------------------------------------------------------
Inkling reports coupling decay strength to the square of the learning rate so the
*steady-state weight norm* stays roughly constant across training horizons. Sketch:
at equilibrium the shrink from decay balances the growth from updates,

    λ · lr · ‖W‖  ≈  lr · ‖update‖ .

For Muon the update is orthogonal, so ‖update‖ is ~constant (independent of ‖W‖).
Balancing the *ratio* of decay-pull to update across an lr schedule that anneals
lr → 0 requires the effective decay per step to scale like lr, i.e. λ_eff = λ·lr,
which applied on top of the lr·(...) step gives an effective decay ∝ lr². We
implement it directly: the decay applied at each step is `wd_base * lr**2`.
"""
from __future__ import annotations
import torch
from torch import Tensor


# ---- Newton–Schulz quintic iteration for the orthogonal factor of a matrix ----
def _zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Return an approximately semi-orthogonal matrix with the same shape as G.

    Coefficients (3.4445, -4.7750, 2.0315) are the standard quintic tuned so the
    iteration pushes all singular values toward 1 from a spectrally-normalised
    start. Runs in bf16/fp16-friendly matmuls. Operates on the last two dims.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    # spectral-normalise so the largest singular value < ~1 (Frobenius upper-bounds it)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    # iterate on the "thin" orientation for efficiency
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


class MuonWithAuxAdam(torch.optim.Optimizer):
    """Muon for the params in group['use_muon']=True, AdamW for the rest.

    Groups are plain param groups; each carries its own hyperparameters. The
    caller builds the groups with `build_param_groups` below.
    """

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            wd_base = group["weight_decay"]        # NOTE: this is the *base*; effective = wd_base*lr^2
            wd_eff = wd_base * (lr ** 2)           # <-- weight decay ∝ lr²

            if group["use_muon"]:
                mom = group["momentum"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    st = self.state[p]
                    if "m" not in st:
                        st["m"] = torch.zeros_like(p)
                    m = st["m"]
                    m.mul_(mom).add_(p.grad)                     # heavy-ball momentum
                    # Nesterov-style lookahead (as in the reference Muon)
                    g = p.grad.add(m, alpha=mom)
                    o = _zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                    # scale so the RMS of the update matches Adam-ish magnitude across shapes
                    scale = max(1.0, p.size(-2) / p.size(-1)) ** 0.5
                    p.mul_(1.0 - wd_eff)                         # decoupled decay ∝ lr²
                    p.add_(o.to(p.dtype), alpha=-lr * scale)
            else:
                # ---- AdamW ----
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    st = self.state[p]
                    if "exp_avg" not in st:
                        st["exp_avg"] = torch.zeros_like(p)
                        st["exp_avg_sq"] = torch.zeros_like(p)
                        st["step"] = 0
                    st["step"] += 1
                    ea, eas = st["exp_avg"], st["exp_avg_sq"]
                    ea.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                    eas.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
                    bc1 = 1 - beta1 ** st["step"]
                    bc2 = 1 - beta2 ** st["step"]
                    denom = (eas.sqrt() / (bc2 ** 0.5)).add_(eps)
                    p.mul_(1.0 - wd_eff)                         # same wd∝lr² coupling
                    p.addcdiv_(ea, denom, value=-lr / bc1)
        return loss


def build_param_groups(model, lr=2e-3, adam_lr=None, weight_decay=0.1,
                       momentum=0.95, betas=(0.9, 0.95), eps=1e-8, ns_steps=5):
    """Split model params into a Muon group (2-D weights) and an AdamW group.

    `lr` is the Muon lr; `adam_lr` defaults to lr/10 (Adam wants a smaller step
    than orthogonalised Muon). `weight_decay` is the *base* λ — the optimiser
    applies λ·lr² each step.
    """
    if adam_lr is None:
        adam_lr = lr * 0.1
    muon_params, adam_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # 2-D matrices go to Muon — EXCEPT embeddings/lm_head (token tables behave
        # like embeddings, not matmul weights) and the router (tiny, Adam is safer).
        is_matrix = (p.ndim == 2)
        is_embed = ("embed" in name) or ("lm_head" in name)
        is_router = ("router" in name) or ("shared_gate" in name) or ("rel_emb" in name)
        if is_matrix and not is_embed and not is_router:
            muon_params.append(p)
        else:
            adam_params.append(p)

    groups = [
        dict(params=muon_params, use_muon=True, lr=lr, weight_decay=weight_decay,
             momentum=momentum, ns_steps=ns_steps),
        dict(params=adam_params, use_muon=False, lr=adam_lr, weight_decay=weight_decay,
             betas=betas, eps=eps),
    ]
    return groups
