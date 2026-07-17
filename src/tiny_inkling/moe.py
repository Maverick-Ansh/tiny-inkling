"""
Mixture-of-Experts layer — the DeepSeek-V3 recipe, faithfully, at small scale.

Three design choices, each annotated where it happens:

  (A) SIGMOID router, not softmax.
      score_i = σ(x · e_i)  is computed *independently per expert*. Unlike a
      softmax gate (which couples all experts through the normaliser), sigmoid
      lets each expert's affinity be judged on its own. Selection & weighting are
      handled separately (below).

  (B) AUX-LOSS-FREE load balancing (the key DeepSeek-V3 trick).
      Classic MoE adds an auxiliary "balance" loss that fights the LM loss and
      distorts gradients. Instead we keep a per-expert BIAS b_i (a buffer, NOT a
      parameter — it receives no gradient) that is added to the score ONLY for
      the top-k *selection*:

          selected = top_k(score_i + b_i)

      After each step we nudge the bias by a fixed amount toward balance:

          b_i ← b_i + u · sign(mean_load − load_i)

      Overloaded experts get their bias lowered (picked less), underloaded ones
      raised. The gate scores that *weight* the outputs are the raw σ scores —
      the bias never touches them, so there is no gradient interference and no
      auxiliary loss term. That is what "auxiliary-loss-free" means.

  (C) JOINT normalisation of selected-routed + shared scores (Inkling detail).
      Shared experts run for every token. We give them their own σ scores and
      normalise the *combined* set {selected routed scores} ∪ {shared scores} to
      sum to 1, then use those as the mixing weights. So a token whose routed
      experts are all lukewarm will lean more on the shared experts, and vice
      versa — the split between "specialist" and "generalist" capacity is learned
      per token, not fixed.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import InklingConfig
from .layers import SwiGLU


class InklingMoE(nn.Module):
    def __init__(self, cfg: InklingConfig):
        super().__init__()
        self.cfg = cfg
        self.n_routed = cfg.n_routed_experts
        self.n_shared = cfg.n_shared_experts
        self.k = cfg.n_active_routed
        self.u = cfg.router_bias_update_speed

        # Router gates: one linear -> per-expert logit, then sigmoid.
        self.router = nn.Linear(cfg.d_model, self.n_routed, bias=False)
        self.shared_gate = nn.Linear(cfg.d_model, self.n_shared, bias=False)

        # Experts. Each is a small SwiGLU. Shared experts are structurally
        # identical; they just always run.
        self.routed_experts = nn.ModuleList(
            [SwiGLU(cfg.d_model, cfg.expert_inter_dim) for _ in range(self.n_routed)]
        )
        self.shared_experts = nn.ModuleList(
            [SwiGLU(cfg.d_model, cfg.expert_inter_dim) for _ in range(self.n_shared)]
        )

        # (B) aux-loss-free balancing bias — a buffer, NOT a Parameter.
        self.register_buffer("expert_bias", torch.zeros(self.n_routed))
        # running load estimate, for logging + the bias update
        self.register_buffer("load_ema", torch.full((self.n_routed,), 1.0 / self.n_routed))
        self.last_stats = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        B, T, D = x.shape
        N = B * T
        flat = x.reshape(N, D)

        # ---- (A) sigmoid affinities (compute in fp32 for stable routing) ----
        logits = self.router(flat).float()                     # (N, n_routed)
        scores = torch.sigmoid(logits)                         # independent gates
        shared_scores = torch.sigmoid(self.shared_gate(flat).float())  # (N, n_shared)

        # ---- (B) top-k selection using score + balancing bias ----
        sel_metric = scores + self.expert_bias.unsqueeze(0)    # bias only for SELECTION
        topk_val, topk_idx = torch.topk(sel_metric, self.k, dim=-1)   # (N, k)
        # gather the *raw* σ scores of the selected experts (bias excluded here)
        topk_scores = scores.gather(-1, topk_idx)              # (N, k)

        # ---- (C) joint normalisation of selected-routed + shared scores ----
        combined = torch.cat([topk_scores, shared_scores], dim=-1)     # (N, k+n_shared)
        denom = combined.sum(-1, keepdim=True).clamp_min(1e-9)
        weights = combined / denom                             # (N, k+n_shared), sums to 1
        routed_w = weights[:, : self.k]                        # (N, k)
        shared_w = weights[:, self.k :]                        # (N, n_shared)

        out = torch.zeros_like(flat)

        # ---- dispatch: run each routed expert on the tokens that selected it ----
        # Loop over experts (n_routed iterations); each processes ~N·k/n_routed tokens.
        # This is the standard "sparse dispatch" — cost is ~k dense-MLP evals/token.
        for e in range(self.n_routed):
            tok, slot = (topk_idx == e).nonzero(as_tuple=True)  # which (token,slot) picked e
            if tok.numel() == 0:
                continue
            ye = self.routed_experts[e](flat[tok])             # (m, D)
            w = routed_w[tok, slot].unsqueeze(-1).to(ye.dtype)
            out.index_add_(0, tok, w * ye)

        # ---- shared experts: always on, weighted by their joint-normalised score ----
        for s in range(self.n_shared):
            ys = self.shared_experts[s](flat)                  # (N, D)
            out = out + shared_w[:, s : s + 1].to(ys.dtype) * ys

        # ---- (B) update the balancing bias (no grad) ----
        if self.training:
            with torch.no_grad():
                # load_i = fraction of the N·k routing slots that went to expert i
                counts = torch.bincount(topk_idx.reshape(-1), minlength=self.n_routed).float()
                load = counts / counts.sum().clamp_min(1.0)
                self.load_ema.mul_(0.99).add_(0.01 * load)
                mean_load = load.mean()
                # nudge bias toward balance; sign(+) => underloaded => raise bias
                self.expert_bias.add_(self.u * torch.sign(mean_load - load))
                # keep bias zero-mean so it can't drift globally
                self.expert_bias.sub_(self.expert_bias.mean())

                # diagnostics: how uneven is the load? (1.0 == perfectly balanced)
                frac_max = load.max() * self.n_routed
                # fraction of experts that received zero tokens this step
                dead = (counts == 0).float().mean()
                self.last_stats = {
                    "load_max_ratio": float(frac_max),
                    "dead_frac": float(dead),
                    "shared_w_mean": float(shared_w.mean()),
                }

        return out.reshape(B, T, D)
