"""GRPO objective + IcePop off-policy correction — the RL math, model-agnostic.

Everything here operates on tensors of per-token log-probs, rewards, and masks,
so the *same* code trains both Qwen (Track B) and Inkling-mini (Track A).

------------------------------------------------------------------------------
GRPO (Group-Relative Policy Optimization), from DeepSeekMath.
------------------------------------------------------------------------------
PPO estimates a per-state *value* V(s) with a learned critic to form the
advantage A = r − V. GRPO throws the critic away. For each prompt it samples a
GROUP of G completions, and uses the group's own reward statistics as the
baseline:

        Aᵢ = (rᵢ − mean_{j∈group} rⱼ) / (std_{j∈group} rⱼ + ε)

Every token in completion i gets that same (whitened) advantage. Intuition: "was
this rollout better or worse than its siblings on the same prompt?" — a baseline
that needs no extra network and is perfectly suited to verifiable rewards, where
a group of rollouts naturally spans success and failure.

The policy-gradient surrogate is PPO's clipped ratio, applied per token:

        ρ_t = π_θ(a_t | s_t) / π_old(a_t | s_t)
        L_t = min( ρ_t·A , clip(ρ_t, 1−ε, 1+ε)·A )
        L   = − (1/Σmask) Σ_t mask_t · L_t          (maximise reward ⇒ minimise −L)

Optionally a per-token KL penalty to a frozen reference keeps us near the base
model (the k3 unbiased estimator, always ≥ 0).

------------------------------------------------------------------------------
IcePop — stabilising ASYNCHRONOUS (off-policy) RL.
------------------------------------------------------------------------------
In async RL the rollouts are produced by a *stale* actor whose weights lag the
learner. So π_old above is not the current policy — it is the behavior policy
that actually generated the tokens, and the two can drift apart between weight
syncs. When they drift, ρ_t explodes or vanishes on some tokens and a handful of
high-variance importance weights wreck the gradient.

IcePop's fix (as in GLM-5's combined objective): treat the importance ratio as a
*trust signal* and **mask out tokens whose ratio leaves a band** [1/c, c]:

        keep_t = 1[ 1/c ≤ ρ_t ≤ c ]

Those tokens are simply dropped from the loss — no gradient from importance
weights we don't trust. This is gentler than clipping (which still leaks a biased
gradient at the clip boundary) and is what lets the learner consume rollouts that
are several updates stale without diverging. We report the fraction masked; if it
climbs, the actor is too stale and should sync more often.
"""
from __future__ import annotations
import torch


def masked_mean(x, mask, dim=None, eps=1e-8):
    mask = mask.to(x.dtype)
    if dim is None:
        return (x * mask).sum() / (mask.sum() + eps)
    return (x * mask).sum(dim) / (mask.sum(dim) + eps)


def group_advantages(rewards: torch.Tensor, group_size: int, eps=1e-6) -> torch.Tensor:
    """rewards: (N,) with N = num_groups * group_size, groups contiguous.
    Returns (N,) whitened advantages (per-group mean 0, std 1)."""
    r = rewards.view(-1, group_size)
    adv = (r - r.mean(dim=1, keepdim=True)) / (r.std(dim=1, keepdim=True) + eps)
    return adv.view(-1)


def kl_k3(logp_new, logp_ref):
    """Unbiased, non-negative KL estimator (Schulman's k3):
        KL ≈ exp(Δ) − Δ − 1,  Δ = logp_ref − logp_new.  Per token."""
    d = logp_ref - logp_new
    return torch.exp(d) - d - 1.0


def grpo_loss(logp_new, logp_old, advantages, resp_mask,
              *, clip_eps=0.2, icepop_c=None, logp_ref=None, kl_beta=0.0):
    """
    logp_new  : (N, T)  log π_θ(a_t|s_t) under the CURRENT policy   (has grad)
    logp_old  : (N, T)  log π_behavior — the policy that generated the tokens (no grad)
    advantages: (N,)    group-relative advantage per sequence
    resp_mask : (N, T)  1 on tokens the policy is responsible for (assistant tokens),
                        0 on prompt / injected <result> / padding
    icepop_c  : float or None. If set, mask tokens with ρ_t outside [1/c, c].

    Returns (loss, stats).
    """
    N, T = logp_new.shape
    adv = advantages.view(N, 1)                      # broadcast advantage over tokens

    log_ratio = logp_new - logp_old                  # log ρ_t
    ratio = torch.exp(log_ratio)

    # ---- PPO clipped surrogate (per token) ----
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    surrogate = torch.minimum(unclipped, clipped)    # (N, T)

    # ---- IcePop trust mask for async staleness ----
    mask = resp_mask
    icepop_frac = torch.tensor(0.0, device=logp_new.device)
    if icepop_c is not None:
        keep = (ratio >= (1.0 / icepop_c)) & (ratio <= icepop_c)
        # fraction of *response* tokens dropped by IcePop
        dropped = resp_mask.bool() & (~keep)
        icepop_frac = dropped.float().sum() / (resp_mask.sum() + 1e-8)
        mask = resp_mask * keep.to(resp_mask.dtype)

    pg_loss = -masked_mean(surrogate, mask)

    # ---- optional KL-to-reference penalty ----
    kl = torch.tensor(0.0, device=logp_new.device)
    if logp_ref is not None and kl_beta > 0:
        kl = masked_mean(kl_k3(logp_new, logp_ref), resp_mask)
    loss = pg_loss + kl_beta * kl

    stats = {
        "pg_loss": float(pg_loss.detach()),
        "kl": float(kl.detach()),
        "ratio_mean": float(masked_mean(ratio, mask).detach()),
        "icepop_masked_frac": float(icepop_frac.detach()),
        "clip_frac": float(masked_mean((unclipped > clipped).float(), resp_mask).detach()),
    }
    return loss, stats


def gather_token_logprobs(logits, input_ids):
    """logits: (N, T, V) for positions predicting input_ids[:,1:].
    Returns per-token logp of the realised next token: (N, T-1)."""
    logp = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
    tgt = input_ids[:, 1:].unsqueeze(-1)
    return logp.gather(-1, tgt).squeeze(-1)
