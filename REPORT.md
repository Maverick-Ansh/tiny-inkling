# tiny-inkling — a small model, explained like a paper

This is the "why", with the math. The code in `src/` and `scripts/` is the "how".
I scaled every dimension down so the whole pipeline fits on 2× Tesla T4, but kept
the **mechanisms** identical to the full recipe, so the derivations below are the
real ones.

Notation: `d` = model width, `H` = query heads, `E` = routed experts, `k` = active
routed experts, `G` = GRPO group size. Vectors are rows; `σ` is the logistic
sigmoid; `⊙` is elementwise product.

---

## 1. The Mixture-of-Experts block

A dense transformer applies the *same* MLP to every token. An MoE layer keeps many
MLPs ("experts") and routes each token to a few. You get the capacity of a huge
network at the FLOPs of a small one, because only `k` of `E` experts fire per token.

We follow **DeepSeek-V3** with three specific choices.

### 1.1 A sigmoid router, not softmax

For token representation `x ∈ ℝ^d` and routed-expert embeddings `e_1..e_E ∈ ℝ^d`,
the affinity for expert `i` is

$$ s_i = \sigma(x \cdot e_i) \in (0,1). $$

The classic router uses `softmax(x·e)`, which *couples* experts: raising one score
lowers the rest through the shared normaliser. With **sigmoid**, each `s_i` is an
independent "does expert *i* want this token?" score. Selection and weighting are
then handled separately, which is what makes the next trick clean.

### 1.2 Auxiliary-loss-**free** load balancing

The failure mode of MoE is collapse: the router sends everything to a few experts,
the rest die. The textbook fix adds an *auxiliary loss* penalising imbalance — but
that loss has a gradient that fights the language-model objective and biases the
router. DeepSeek-V3's insight: **balance the routing, not the loss.**

Keep a per-expert bias `b_i` that is **not a parameter** (no gradient). It enters
*only the top-k selection*, never the weights:

$$ \text{selected}(x) = \operatorname*{top\text{-}k}_i\; (s_i + b_i). $$

After each step, look at the realised load `ℓ_i` = fraction of routing slots that
went to expert `i`, and nudge the bias toward the mean load `\bar{ℓ}` by a fixed
step `u`:

$$ b_i \leftarrow b_i + u \cdot \operatorname{sign}(\bar{ℓ} - ℓ_i). $$

Overloaded experts (`ℓ_i > \bar ℓ`) get their bias lowered and are picked less;
starved experts get raised. It's a control loop, not a loss term — so the LM
gradient is untouched and there is *no* auxiliary objective. (`src/tiny_inkling/moe.py`,
the `expert_bias` buffer.) In our smoke test the load's max/mean ratio sat around
1.2–1.3 with **zero dead experts**, confirming the loop balances.

The weights that actually mix the experts are the *raw* `s_i` — the bias is
selection-only, so it can never distort the combination.

### 1.3 Joint normalisation of routed + shared experts

Two **shared** experts run for every token (always-on generalists). Inkling gives
them their own sigmoid scores `s^{sh}_j` and normalises the *combined* set of the
selected-routed and shared scores to form the mixing weights:

$$
w = \frac{[\,s_{i_1},\dots,s_{i_k},\; s^{sh}_1,\dots,s^{sh}_m\,]}
         {\sum s_{i} + \sum s^{sh}_j},
\qquad
y = \sum_{t=1}^{k} w_t \,\mathrm{FFN}_{i_t}(x) \;+\; \sum_{j=1}^{m} w_{k+j}\,\mathrm{FFN}^{sh}_j(x).
$$

So the split between specialist and generalist capacity is *learned per token*: if a
token's chosen experts are all lukewarm, its shared-expert weight rises
automatically. (DeepSeek-V3 adds shared experts with fixed weight; the joint-norm is
the Inkling twist, and it's what `combined = cat([topk_scores, shared_scores])` does.)

---

## 2. Attention

### 2.1 Sliding-window / global interleave (5:1) + GQA

Full attention is `O(T²)`. Most layers don't need it: local structure dominates. So
5 of every 6 layers attend only within a window `W` (`O(T·W)`), and every 6th is
global. With 6 layers we get exactly one global layer — the 5:1 ratio, exactly.

**Grouped-query attention**: `H` query heads but only `H_kv < H` key/value heads,
each KV head shared by `H/H_kv` query heads. The KV cache — the thing that bounds
long-context inference memory — shrinks by `H/H_kv` (here 6/2 = 3×).

### 2.2 Shaw relative position embeddings instead of RoPE

RoPE rotates queries/keys by an angle proportional to absolute position. It's great
but its extrapolation past the trained length is fragile. We use **Shaw et al.
(2018)**: a learned bias indexed by the *relative* offset `j − i`, clipped to `±c`:

$$
\text{logit}_{ij} = \frac{q_i \cdot k_j}{\sqrt{d}} + \frac{q_i \cdot r_{\,\mathrm{clip}(j-i,\,-c,\,c)}}{\sqrt{d}},
$$

where `r_{-c..c}` is a learned table of `2c+1` vectors. Two things to notice:

- The second term is `q_i · r_δ` — a *content-dependent* positional bias (the query
  decides how much a given relative offset matters), computed efficiently by one
  `q·rᵀ` product then gathered by offset (the *Music Transformer* re-indexing trick;
  see `_rel_index` in `attention.py`).
- **Clipping is the extrapolation mechanism.** Any offset beyond `c` reuses the
  boundary embedding `r_{±c}`, so a model trained at `T=512` doesn't see out-of-range
  positions at `T=2048` — it just sees "far". Our smoke test forwards `2×` the
  trained length with finite logits and no code changes.

### 2.3 Short convolutions at two points

A depthwise **causal** conv (kernel 4) — one tiny filter per channel, seeing only
`t−3..t` — is inserted:

1. on **K and V** right after their projections, and
2. on the **attention/MLP branch outputs** before they rejoin the residual stream.

Depthwise ⇒ negligible params/FLOPs; causal ⇒ no future leakage (left-pad by `k−1`).
It gives every token a cheap, learned, short-range temporal mixing that attention
would otherwise have to spend capacity on (local n-gram / smoothing features). We
initialise the kernel to a near-identity (last tap ≈ 1) so each block *starts* as a
normal transformer block and *learns* to use the conv — which keeps early training
stable. (`ShortConv` in `layers.py`.)

---

## 3. Optimisation: Muon + Adam, and weight decay ∝ lr²

### 3.1 Muon for the matrices

Adam adapts a per-coordinate step from gradient moments. **Muon** treats a weight
*matrix* `W` and its momentum `M` as a matrix and takes an **orthogonalised** step.
If `M = UΣVᵀ` is the SVD, the ideal update direction is `UVᵀ` (all singular values
set to 1) — it equalises the update spectrum so no direction dominates. Computing an
SVD every step is too slow, so Muon uses the quintic **Newton–Schulz** iteration on a
spectrally-normalised start `X_0 = M/‖M‖_F`:

$$ X \leftarrow a X + b (XX^\top)X + c (XX^\top)^2 X,\quad (a,b,c)=(3.4445,-4.7750,2.0315), $$

which converges to the orthogonal factor using only matmuls (fp16-friendly on T4).
Empirically this lets the big matmul weights take much larger, better-conditioned
steps than Adam. We keep **AdamW** for embeddings, norms, conv kernels, the router,
and the relative-position table — orthogonalisation is only meaningful for matmul
matrices, and those small/structured params behave better under Adam.
(`src/tiny_inkling/muon.py`.)

### 3.2 Coupling weight decay to lr²

Inkling reports coupling decay to the **square of the learning rate**, which keeps
the steady-state weight norm stable across training horizons. The argument:

At equilibrium the shrink from decoupled weight decay balances the growth from
updates. For a decoupled decay `λ_eff` and update magnitude `‖Δ‖`,

$$ \lambda_{\text{eff}}\,\lVert W\rVert \;\approx\; \lVert \Delta \rVert. $$

Muon's update is **orthogonal**, so `‖Δ‖` is roughly *constant* (set by `lr`, not by
`‖W‖`). To keep the *ratio* of decay-pull to update invariant as an lr schedule
anneals `lr → 0`, the effective decay must scale with `lr`. Since we apply it on top
of the `lr·(…)` step, the decay coefficient we set is `λ·lr`, giving an **effective
decay ∝ lr²**. In code: `wd_eff = wd_base * lr**2`, applied as `W ← (1 − wd_eff)·W`
for *both* the Muon and Adam groups. This is the "modular-manifold"-flavoured trick
that keeps weight norms from drifting as you change the total step count.

---

## 4. Agentic RL — the centerpiece

This is the part worth understanding deeply, because it's how a model learns to
*act*, not just predict.

### 4.1 What "agentic" means here, and why the reward is *verifiable*

An **agentic** task requires the model to take **actions** — call a tool, read the
result, decide the next action — over **multiple turns**, and it is judged only on
whether the **final outcome is correct**. There are no labels on the intermediate
reasoning. The teacher is an **environment** that (a) hands out a task and (b) can
*check* the answer. This "verifiable reward" is the crux: it's cheap, unhackable
(you can't fake `3847×291`), and infinitely generable. Large-scale agentic RL is
exactly this at scale — synthetic and human-authored environments with checkers.

Our two environments (`scripts/envs.py`):

- **CalcEnv** — arithmetic the model can't do reliably in-weights, so it must
  *offload* to a `calc(...)` tool. Reward shaped as: +0.15 valid tool call, +0.15
  tool used *correctly*, +0.70 final `<answer>` matches, −0.20 for guessing without
  the tool. Shaping matters — it gives partial-credit gradient before the model ever
  gets a full answer right.
- **LookupEnv** — a per-task knowledge base needs **multi-hop** tool use:
  `lookup(person)→pet`, then `lookup(pet)→city`, then answer. Rewards chaining.

The environment drives the loop: it reads the model's `<tool>…</tool>`, executes it,
and injects `<result>…</result>`; generation resumes until `<answer>…</answer>`.

### 4.2 GRPO: advantages without a value network

PPO needs a learned **critic** `V(s)` to compute advantages `A = r − V`. For a
verifiable-reward setting that critic is wasteful. **GRPO** (DeepSeekMath) replaces
it with the statistics of a *group* of `G` rollouts sampled from the same prompt:

$$ A_i = \frac{r_i - \operatorname{mean}_{j}(r_j)}{\operatorname{std}_j(r_j) + \varepsilon},\qquad j \in \text{group}. $$

Every token of rollout `i` inherits that whitened advantage. The baseline is "how
did this rollout do versus its siblings on the *same* prompt?" — perfectly matched
to verifiable rewards, where a group naturally spans successes and failures. (We saw
this live: a lookup group scored 0.25–1.0 because one rollout typo'd `lookup(gziggy)`
while others nailed it — that spread *is* the learning signal.)

The policy update is PPO's clipped surrogate, per token, with importance ratio
`ρ_t = π_θ(a_t|s_t)/π_{old}(a_t|s_t)`:

$$
\mathcal{L} = -\,\mathbb{E}_t\Big[\min\big(\rho_t A,\; \operatorname{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\,A\big)\Big] \;+\; \beta\, \mathrm{KL}\!\left(\pi_\theta \,\|\, \pi_{\text{ref}}\right),
$$

averaged over the **response tokens only** (prompt and injected `<result>` are
masked). The KL to a frozen reference (the base model, via `disable_adapter()`) keeps
the policy from wandering off into gibberish that games the reward. We use the
non-negative **k3** estimator `KL ≈ e^{Δ} − Δ − 1`, `Δ = logπ_ref − logπ_θ`.
(`scripts/grpo.py`.)

### 4.3 Why *asynchronous*, and how IcePop keeps it stable

At scale you don't want the (expensive) generation and the (cheap) gradient step to
wait on each other. So you make them **asynchronous**: an **actor** keeps generating
rollouts with a *snapshot* of the weights while a **learner** updates continuously,
and the actor's snapshot is refreshed only every `K` steps. We implement exactly this
with a background actor thread on GPU 1 and the learner on GPU 0 (`scripts/rl_agentic.py`).

The catch: by the time the learner consumes a rollout, the policy has moved — the
data is **off-policy / stale**. The importance ratio `ρ_t` is what corrects for this
(that's why `π_old` in §4.2 is the *behavior* policy that actually generated the
tokens, not the current one). But when the policy drifts, some `ρ_t` explode and a
handful of high-variance importance weights wreck the gradient.

**IcePop** (from the GLM-5 recipe) treats the ratio as a *trust signal* and **masks
out tokens whose ratio leaves a band**:

$$ \text{keep}_t = \mathbf{1}\!\left[\tfrac{1}{c} \le \rho_t \le c\right], $$

dropping untrusted tokens from the loss entirely — gentler than clipping, which still
leaks a biased gradient at the boundary. This is what lets the learner safely eat
rollouts that are several updates stale. We log the **masked fraction** and the
**staleness** (updates since the actor's snapshot); if the masked fraction climbs,
the actor is too stale and should sync more often. In a unit test, IcePop masked 0%
of tokens on-policy and ~81% under injected staleness — exactly the intended
behaviour.

> The combined objective — GRPO group-advantages + token-level PPO clip + IcePop
> trust mask + KL-to-reference — is the same shape as GLM-5's Eq. 1, just at 0.5B
> scale with two environments.

### 4.4 The agent loop, concretely

```
<user> What is 9721 * 569? </user>
<assistant> <tool>calc(9721*569)</tool>        ← policy acts, generation stops at </tool>
<result> 5531249 </result>                     ← ENV executes + injects (masked in loss)
<assistant> <answer>5531249</answer>           ← policy answers, verify() → reward 1.0
```

Only the `<assistant>` tokens carry loss; the `<result>` the environment produced is
masked, because the policy isn't responsible for it.

---

## 5. Results

Runs on Kaggle 2×T4, fp16. Plots in `assets/`, raw metrics in the checkpoint dirs'
`*_log.jsonl`.

### Track A — Inkling-mini (from scratch)
- **Model:** 57.6M parameters total, **19.3M active/token** (top-6 of 32 experts +
  2 shared, per MoE layer). ~3× the "dense-equivalent-of-active" capacity for the
  active FLOPs.
- **Pretraining (TinyStories, ~460 steps ≈ 30M tokens in the time-box):** train loss
  **9.10 → 3.25** (val 3.25). Steady, stable descent under Muon+Adam with the wd∝lr²
  coupling — no loss spikes in fp16.
- **Aux-loss-free balancing works:** the max-expert-load / mean-load ratio fell from
  **3.13 at init to ≈1.18–1.25**, with **zero dead experts** throughout — the bias
  control loop balances the experts *without* any auxiliary loss term.
- **RelPos extrapolation:** forwarding at 2× the trained length produces finite,
  well-behaved logits with no code change (Shaw clipping, verified in the smoke test).
- **Generation:** at loss 3.25 the model produces on-topic TinyStories vocabulary and
  story scaffolding ("Once upon a time,", named characters, park/dog/tree motifs) but
  not yet fluent syntax — honestly, that's an undertraining artifact of the few-hour,
  MoE-python-loop-bottlenecked budget, not an architectural limit; the loss was still
  falling when we stopped to hand the GPUs to the RL centerpiece.

### Track B — agentic RL on Qwen2.5-0.5B (the centerpiece)
*(reward / accuracy / tool-use curves + IcePop/staleness traces — see below and
`assets/rl_reward.png`, `assets/rl_offpolicy.png`; before/after tool-use table from
`eval_agentic.py`.)*

We validated the loop qualitatively before the full run: with one-shot format
priming the base model already chains tools correctly on easy cases
(`lookup(dave)→milo→lookup(milo)→delhi→<answer>delhi</answer>`), while harder cases
fail on argument typos (`lookup(gziggy)`), giving the within-group reward spread
(0.25–1.0) that GRPO turns into advantage.

### Bonsai ternary
*(size and perplexity trade-off — `assets/bonsai_ternary.png`, numbers in
`assets/bonsai_ternary.json`.)*

## 6. Bonsai 1.58-bit ternary quantization

Post-training absmean ternary quant `{-1,0,+1}·γ`, `γ = mean(|W|)`, on the matmul
weights (embeddings/norms/router kept fp16). "1.58 bit" = `log₂3`. This is the honest,
no-retraining cost: how small, how much worse. Numbers in `assets/bonsai_ternary.json`.

## 7. How it ran (2× T4, free tier)

- **fp16 forced** — T4 (Turing) has no bf16 tensor cores. Autocast fp16 + GradScaler;
  RMSNorm/softmax reductions in fp32 for stability.
- **DDP** across both T4s for pretrain (`find_unused_parameters=True` because MoE
  leaves unselected experts gradient-less each step).
- **Checkpoint/resume via the HF Hub** every few minutes and on a time-budget exit,
  so a Kaggle session reset resumes exactly where it stopped (optimiser, scaler,
  step, and RNG all restored).
- The RL centerpiece uses **both GPUs at once**: learner on GPU 0, async actor on
  GPU 1.

---

*Reading order for the code: `config.py` → `layers.py` → `attention.py` → `moe.py`
→ `model.py` → `muon.py`, then `envs.py` → `grpo.py` → `rl_agentic.py`.*
