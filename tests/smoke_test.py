"""Fast CPU smoke test: build the model, run fwd/bwd, take an optimiser step,
and verify the MoE routing + balancing bias actually update. Run:  python tests/smoke_test.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch
from tiny_inkling import InklingForCausalLM, DEBUG, build_param_groups, MuonWithAuxAdam

torch.manual_seed(0)
cfg = DEBUG
model = InklingForCausalLM(cfg)
print(f"total params      : {model.num_params():,}")
print(f"active params/tok : {model.num_params(only_active=True):,}")

B, T = 4, cfg.max_seq_len
ids = torch.randint(0, cfg.vocab_size, (B, T))
labels = ids.clone()

# capture the balancing bias at init (all zeros) BEFORE any forward — the
# aux-loss-free update happens *inside* forward() when training, not in opt.step()
model.train()
bias_before = model.blocks[-1].ffn.expert_bias.clone()

# forward + loss (this is the forward that should nudge the balancing bias)
out = model(ids, labels=labels)
loss0 = out["loss"].item()
print(f"initial loss      : {loss0:.4f}  (random-init ~ ln(vocab)={torch.log(torch.tensor(float(cfg.vocab_size))):.3f})")
assert torch.isfinite(out["loss"]), "loss is not finite!"

opt = MuonWithAuxAdam(build_param_groups(model, lr=2e-3))
out["loss"].backward()
# sanity: gradients exist on a Muon matrix and an Adam vector
assert model.blocks[-1].ffn.routed_experts[0].w_gate.weight.grad is not None
assert model.embed.weight.grad is not None
opt.step()
opt.zero_grad()

bias_after = model.blocks[-1].ffn.expert_bias
moved = (bias_after - bias_before).abs().sum().item()
print(f"balancing-bias Δ  : {moved:.5f}  (should be > 0 => aux-loss-free update fired)")
assert moved > 0, "expert balancing bias did not update!"

# a few steps should reduce loss on this tiny fixed batch (overfit check)
for i in range(20):
    out = model(ids, labels=labels)
    out["loss"].backward()
    opt.step(); opt.zero_grad()
print(f"loss after 20 steps: {out['loss'].item():.4f}  (should be << {loss0:.3f})")
assert out["loss"].item() < loss0, "model did not learn on the overfit batch!"

# MoE diagnostics
print("moe stats         :", model.moe_stats())

# extrapolation sanity: forward a longer sequence than trained (RelPos should not crash)
long_ids = torch.randint(0, cfg.vocab_size, (1, T * 2))
with torch.no_grad():
    lo = model(long_ids)
assert torch.isfinite(lo["logits"]).all(), "RelPos extrapolation produced non-finite logits!"
print(f"extrapolation ok  : forwarded T={T*2} (trained T={T}), logits finite")
print("\nSMOKE TEST PASSED ✅")
