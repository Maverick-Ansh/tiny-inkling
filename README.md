# tiny-inkling

A small, **heavily-annotated** re-implementation of an *Inkling*-style
Mixture-of-Experts transformer (the recipe closely follows **DeepSeek-V3**), plus
the part I actually care about: an **asynchronous GRPO agentic-RL** post-training
loop that teaches a model to *use tools*. Everything is scaled to run end-to-end
on **2× Tesla T4** (Kaggle, free tier) in a few hours, with checkpoint/resume so
it survives session resets.

> This repo is written to be *read*. Every non-obvious line has a comment
> explaining the math or the design choice. If you want the derivations, see
> **[REPORT.md](REPORT.md)** — it's written like a short paper.

---

## Two tracks

| | Track A — **Inkling-mini** | Track B — **Agentic RL** |
|---|---|---|
| Goal | Prove the *architecture* learns | Prove the *agentic RL loop* works — visibly |
| Model | ~70M-param MoE, **built from scratch** here | Qwen2.5-0.5B (a real small foundation model) |
| Data | TinyStories | Synthetic **verifiable** tool-use environments |
| Signal | loss ↓, experts balance, RelPos extrapolates | tool-call accuracy ↑ under RL |
| Why split? | A from-scratch 70M model barely knows English, so its *agentic* competence is toy-level. We showcase the architecture on Track A and get **real, visible agentic gains** on Track B — the same async-GRPO code drives both. |

---

## The architecture (Track A) — what's different from a vanilla transformer

Every one of these is a deliberate Inkling/DeepSeek-V3 choice. Scaled-down numbers
in parentheses.

- **MoE with a sigmoid router + aux-loss-free load balancing.**
  256→**32** routed experts, **2** shared experts, top-**6** routed active per token.
  The router uses **sigmoid** gates (independent per expert, not softmax). Load is
  balanced by a **bias that receives no gradient** and is nudged toward balance
  each step — *no auxiliary loss fighting the LM objective*. Selected-routed and
  shared scores are **normalised jointly** to form the mixing weights.
  → [`src/tiny_inkling/moe.py`](src/tiny_inkling/moe.py)

- **Interleaved sliding-window / global attention at 5:1**, with **GQA** (2 KV heads).
  → [`src/tiny_inkling/attention.py`](src/tiny_inkling/attention.py)

- **Shaw (2018) relative position embeddings instead of RoPE.** A learned,
  distance-**clipped** bias added to attention logits. Clipping is what gives
  length **extrapolation** (train at 512, run at 2×). (See also *Music Transformer*
  for the efficient indexing.) → [`attention.py`](src/tiny_inkling/attention.py)

- **Short convolutions at two points** (depthwise, causal): on **K and V** after
  their projections, and on the **attention/MLP branch outputs** before they
  rejoin the residual stream. → [`layers.py`](src/tiny_inkling/layers.py) + [`model.py`](src/tiny_inkling/model.py)

- **Hybrid optimiser: Muon for 2-D matrices, AdamW for the rest**, with
  **weight decay coupled to lr²** (keeps the steady-state weight norm stable across
  training horizons — a modular-manifold-flavoured trick).
  → [`src/tiny_inkling/muon.py`](src/tiny_inkling/muon.py)

---

## The centerpiece (Track B) — how you train "agentic"

Agentic training = the model learns to **call tools and act over multiple turns**,
rewarded only by whether the final answer is **correct** (a *verifiable* reward).
No human labels on the reasoning — just an environment that checks the result.

The loop, in one breath:

1. **Environments** produce a task (e.g. "what is 3847 × 291?") and a **verifier**.
   → [`scripts/envs.py`](scripts/envs.py)
2. The **actor** (on GPU 1) samples a *group* of G rollouts per task. A rollout is
   a multi-turn transcript with `<tool>…</tool>` calls; the environment executes
   the tool and returns `<result>…</result>`; the model continues to a final answer.
3. The **verifier** scores each rollout (correct answer, valid tool syntax, brevity).
4. **GRPO** turns the group's rewards into advantages by *whitening within the group*
   (no learned value function), and the **learner** (on GPU 0) updates the policy.
5. Because the actor runs **asynchronously** with slightly **stale** weights, we
   apply an **importance-ratio correction with IcePop-style masking** to stay
   stable off-policy. → [`scripts/rl_agentic.py`](scripts/rl_agentic.py)

The math (GRPO objective, the group-advantage estimator, and the async off-policy
correction) is derived in **[REPORT.md §RL](REPORT.md)**.

---

## Reproduce

```bash
# 0) deps
pip install -e .

# 1) sanity (CPU, ~10s): builds the model, checks routing/balancing/extrapolation
python tests/smoke_test.py

# 2) Track A — tokenizer, data, pretrain on 2×T4 (checkpointed, resumable)
python scripts/train_tokenizer.py
python scripts/prepare_data.py
python scripts/pretrain.py            # DDP over both T4s, fp16
python scripts/sft.py                 # teach the tool-call format

# 3) Track B — agentic RL (the point)
python scripts/rl_agentic.py --base Qwen/Qwen2.5-0.5B-Instruct

# 4) stretch — Bonsai-style ternary {-1,0,+1} quantization of Inkling-mini
python scripts/quantize_bonsai.py
```

On Kaggle everything is orchestrated from a notebook that toggles the GPUs to
avoid idle-timeout and pushes checkpoints to GitHub/HF via Kaggle Secrets, so a
session reset costs nothing (see `notebook/` once training starts).

---

## Trained artifacts (Hugging Face Hub)

Checkpoints are pushed here every few minutes during training, which is also the
resume source after a Kaggle session reset:

| repo | what |
|---|---|
| [`AnshVivek/tiny-inkling-pretrain`](https://huggingface.co/AnshVivek/tiny-inkling-pretrain) | Inkling-mini pretrained on TinyStories (`latest.pt` = model+opt+scaler+step) |
| [`AnshVivek/tiny-inkling-sft`](https://huggingface.co/AnshVivek/tiny-inkling-sft) | Inkling-mini after tool-format SFT |
| [`AnshVivek/tiny-inkling-rl-qwen`](https://huggingface.co/AnshVivek/tiny-inkling-rl-qwen) | Qwen2.5-0.5B agentic-RL LoRA adapter |

## Status

This is a live build. See the commit history and `REPORT.md` for what's landed.
Runs on Kaggle's free 2×T4; fp16 is **forced** (T4 has no bf16).

## Credits / reading
- DeepSeek-V3 (MoE, aux-loss-free balancing, sigmoid router)
- Shaw et al. 2018, *Self-Attention with Relative Position Representations*
- Huang et al. 2018, *Music Transformer* (efficient relative attention)
- Jordan et al. 2024, *Muon* optimizer
- GRPO (DeepSeekMath) + IcePop-style off-policy masking for async RL

*Built as a learning project — small scale, real mechanisms.*
