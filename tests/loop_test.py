"""CPU test of the training-loop plumbing: BinDataset, cosine_lr, CheckpointManager
save/resume, and a few real optimiser steps on DEBUG config with a synthetic bin."""
import sys, os, tempfile, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import numpy as np, torch
from tiny_inkling import InklingForCausalLM, DEBUG, build_param_groups, MuonWithAuxAdam
from common import BinDataset, cosine_lr, CheckpointManager, set_seed

d = tempfile.mkdtemp()
# synthetic uint16 token stream
toks = np.random.randint(0, DEBUG.vocab_size, size=200_000, dtype=np.uint16)
toks.tofile(os.path.join(d, "train.bin"))
ds = BinDataset(os.path.join(d, "train.bin"), DEBUG.max_seq_len)
x, y = ds.batch(4, "cpu")
assert x.shape == (4, DEBUG.max_seq_len) and y.shape == x.shape
assert (x[:, 1:] == y[:, :-1]).all(), "label shift is wrong!"
print("BinDataset ok, shapes", x.shape)

# cosine schedule monotonic sanity
lrs = [cosine_lr(s, warmup=10, total=100, base_lr=1e-3) for s in range(100)]
assert lrs[0] < lrs[10] and lrs[10] >= lrs[99], "cosine schedule shape wrong"
print("cosine_lr ok: warmup peak", round(max(lrs), 6), "end", round(lrs[-1], 6))

set_seed(0)
model = InklingForCausalLM(DEBUG)
opt = MuonWithAuxAdam(build_param_groups(model, lr=2e-3))
model.train()
for step in range(6):
    lr = cosine_lr(step, warmup=2, total=6, base_lr=2e-3)
    for g in opt.param_groups:
        g["lr"] = lr if g["use_muon"] else lr * 0.1
    x, y = ds.batch(4, "cpu")
    out = model(x, labels=y)
    out["loss"].backward()
    opt.step(); opt.zero_grad()
print("6 steps ok, loss", round(out["loss"].item(), 3))

# checkpoint save + resume round-trip (local only, no HF)
cm = CheckpointManager(d, hf_repo=None, is_main=True)
cm.save(dict(model=model.state_dict(), opt=opt.state_dict(),
             scaler={}, step=6, cfg=DEBUG.__dict__), push=False)
st = cm.load_latest()
assert st is not None and st["step"] == 6
model2 = InklingForCausalLM(DEBUG)
model2.load_state_dict(st["model"])
# params identical after resume
for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
    assert torch.equal(p1, p2), f"resume mismatch at {n1}"
print("checkpoint save/resume ok, step", st["step"])
print("\nLOOP TEST PASSED ✅")
