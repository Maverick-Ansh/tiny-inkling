"""Bonsai-style 1.58-bit (ternary) quantization of Inkling-mini — a side-experiment.

"1.58-bit" is log2(3): every weight becomes one of three values {-1, 0, +1} times a
per-tensor scale (this is BitNet b1.58's *absmean* scheme). We apply it
**post-training** (no quantization-aware fine-tuning) to the big matmul weights and
measure the honest cost: how much smaller does the model get, and how much worse is
its validation perplexity?

    absmean quant:   γ = mean(|W|)                       (per-tensor scale)
                     W_q = clip(round(W / γ), -1, +1)     (ternary integers)
                     Ŵ  = γ · W_q                         (dequantized for eval)

Embeddings / norms / router are kept in fp16 (BitNet keeps the token tables and
tiny params high-precision — quantizing them hurts far more than it saves). We
report the *effective* size assuming ternary weights are bit-packed at 1.58 b/w.
"""
import os, sys, json, argparse, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch
from tiny_inkling import InklingForCausalLM, InklingConfig
from common import BinDataset, CheckpointManager


def ternarize_(w: torch.Tensor):
    """In-place absmean ternary quant → dequant. Returns (num_weights, scale)."""
    gamma = w.abs().mean().clamp_min(1e-8)
    q = (w / gamma).round().clamp_(-1, 1)
    w.copy_(q * gamma)
    return w.numel(), float(gamma)


@torch.no_grad()
def val_ppl(model, val, device, n=40, bs=16, seq=512):
    model.eval()
    tot = 0.0
    for _ in range(n):
        x, y = val.batch(bs, device)
        with torch.autocast("cuda", dtype=torch.float16):
            tot += float(model(x, labels=y)["loss"])
    return math.exp(tot / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_local", default="/kaggle/working/checkpoints/pretrain/latest.pt")
    ap.add_argument("--ckpt_repo", default="AnshVivek/tiny-inkling-pretrain")
    ap.add_argument("--data", default="/kaggle/working/data")
    ap.add_argument("--out", default="/kaggle/working/tiny-inkling/assets")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda:0"

    cm = CheckpointManager(os.path.dirname(args.ckpt_local), hf_repo=args.ckpt_repo)
    state = cm.load_latest(name=os.path.basename(args.ckpt_local))
    assert state is not None, "no checkpoint to quantize"
    cfg = InklingConfig(**state["cfg"])
    model = InklingForCausalLM(cfg).to(device)
    model.load_state_dict(state["model"])

    val = BinDataset(os.path.join(args.data, "val.bin"), cfg.max_seq_len)
    ppl_fp16 = val_ppl(model, val, device)

    # ---- ternarize the matmul weights (skip embeddings/norms/router) ----
    n_tern, n_kept = 0, 0
    for name, p in model.named_parameters():
        keep = (p.ndim != 2) or ("embed" in name) or ("lm_head" in name) \
            or ("router" in name) or ("shared_gate" in name) or ("rel_emb" in name)
        if keep:
            n_kept += p.numel()
        else:
            cnt, _ = ternarize_(p.data)
            n_tern += cnt
    ppl_tern = val_ppl(model, val, device)

    # ---- size accounting ----
    fp16_bytes = (n_tern + n_kept) * 2
    tern_bytes = n_tern * (math.log2(3) / 8) + n_kept * 2   # 1.58 b/w packed + fp16 rest
    res = dict(
        params_total=n_tern + n_kept, params_ternarized=n_tern, params_kept_fp16=n_kept,
        ppl_fp16=round(ppl_fp16, 3), ppl_ternary=round(ppl_tern, 3),
        ppl_increase_pct=round(100 * (ppl_tern - ppl_fp16) / ppl_fp16, 1),
        size_fp16_mb=round(fp16_bytes / 1e6, 2), size_ternary_mb=round(tern_bytes / 1e6, 2),
        compression_x=round(fp16_bytes / tern_bytes, 2),
    )
    with open(os.path.join(args.out, "bonsai_ternary.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))

    # ---- plot ----
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
        ax[0].bar(["fp16", "ternary\n(1.58b)"], [res["size_fp16_mb"], res["size_ternary_mb"]],
                  color=["#4C72B0", "#55A868"])
        ax[0].set_ylabel("model size (MB)"); ax[0].set_title(f"{res['compression_x']}× smaller")
        ax[1].bar(["fp16", "ternary\n(1.58b)"], [res["ppl_fp16"], res["ppl_ternary"]],
                  color=["#4C72B0", "#C44E52"])
        ax[1].set_ylabel("val perplexity"); ax[1].set_title(f"+{res['ppl_increase_pct']}% ppl (post-training)")
        fig.suptitle("Bonsai-style 1.58-bit ternary quantization of Inkling-mini")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "bonsai_ternary.png"), dpi=120)
        print("saved plot ->", os.path.join(args.out, "bonsai_ternary.png"))
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
