"""Render the figures for the README/REPORT from the JSONL training logs.

Reads whatever logs exist and writes PNGs into assets/:
  * pretrain_loss.png        — train loss (+ val loss markers)
  * pretrain_balance.png     — MoE load-max/mean ratio over training (balancing at work)
  * rl_reward.png            — agentic RL: reward / accuracy / tool-use over steps
  * rl_offpolicy.png         — IcePop masked-fraction and staleness over steps
  * rl_effort.png            — the effort dial: per-effort generated tokens (should
                               fan out: low compresses, high stays roomy) + accuracy
"""
import os, sys, json, glob, argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    for line in open(path):
        line = line.strip()
        if line:
            try: rows.append(json.loads(line))
            except Exception: pass
    return rows


def ema(xs, a=0.1):
    out, m = [], None
    for x in xs:
        m = x if m is None else (1 - a) * m + a * x
        out.append(m)
    return out


def plot_pretrain(logdir, outdir):
    rows = load_jsonl(os.path.join(logdir, "train_log.jsonl"))
    steps = [r for r in rows if "loss" in r and "event" not in r]
    if steps:
        xs = [r["step"] for r in steps]; ys = [r["loss"] for r in steps]
        plt.figure(figsize=(7, 4))
        plt.plot(xs, ys, alpha=0.3, color="#4C72B0", label="train loss")
        plt.plot(xs, ema(ys), color="#1f3a66", label="train loss (EMA)")
        val = [r for r in rows if "val_loss" in r]
        if val:
            plt.scatter([r["step"] for r in val], [r["val_loss"] for r in val],
                        color="#C44E52", zorder=5, label="val loss", s=25)
        plt.xlabel("step"); plt.ylabel("cross-entropy loss")
        plt.title("Inkling-mini pretraining (TinyStories, 2×T4 fp16)")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(outdir, "pretrain_loss.png"), dpi=120); plt.close()
        print("wrote pretrain_loss.png")

        if any("load_max_ratio" in r for r in steps):
            xs = [r["step"] for r in steps if "load_max_ratio" in r]
            ys = [r["load_max_ratio"] for r in steps if "load_max_ratio" in r]
            plt.figure(figsize=(7, 4))
            plt.plot(xs, ys, alpha=0.35, color="#55A868")
            plt.plot(xs, ema(ys), color="#2f6b3f", label="load max/mean (EMA)")
            plt.axhline(1.0, ls="--", color="gray", label="perfect balance (1.0)")
            plt.xlabel("step"); plt.ylabel("max expert load / mean load")
            plt.title("Aux-loss-free load balancing (lower → more balanced)")
            plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
            plt.savefig(os.path.join(outdir, "pretrain_balance.png"), dpi=120); plt.close()
            print("wrote pretrain_balance.png")


def plot_rl(logdir, outdir):
    rows = load_jsonl(os.path.join(logdir, "rl_log.jsonl"))
    if not rows:
        return
    xs = [r["step"] for r in rows]
    plt.figure(figsize=(7, 4))
    for key, col in [("reward", "#4C72B0"), ("acc", "#C44E52"), ("tool_use", "#55A868")]:
        if any(key in r for r in rows):
            ys = [r.get(key, float("nan")) for r in rows]
            plt.plot(xs, ema(ys, 0.08), color=col, label=f"{key} (EMA)")
    plt.xlabel("learner step"); plt.ylabel("value"); plt.ylim(-0.1, 1.05)
    plt.title("Async GRPO agentic RL — Qwen2.5-0.5B")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "rl_reward.png"), dpi=120); plt.close()
    print("wrote rl_reward.png")

    plt.figure(figsize=(7, 4))
    if any("icepop" in r for r in rows):
        plt.plot(xs, [r.get("icepop", 0) for r in rows], color="#8172B3", label="IcePop masked frac")
    if any("staleness" in r for r in rows):
        ax2 = plt.gca().twinx()
        ax2.plot(xs, [r.get("staleness", 0) for r in rows], color="#CCB974", alpha=0.7, label="staleness (updates)")
        ax2.set_ylabel("staleness (learner updates behind)")
    plt.xlabel("learner step"); plt.gca().set_ylabel("IcePop masked fraction")
    plt.title("Off-policy correction under async staleness")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "rl_offpolicy.png"), dpi=120); plt.close()
    print("wrote rl_offpolicy.png")

    # ---- the effort dial: per-effort token spend (top) and accuracy (bottom) ----
    # Each step only logs the efforts its sampled groups drew, so each line has its
    # own x-axis. EMA over ~30 points smooths the group-to-group task variance.
    EFF_ORDER = ["low", "medium", "high", "none"]
    EFF_COLOR = {"low": "#C44E52", "medium": "#DD8452", "high": "#4C72B0", "none": "#8172B3"}
    if any(r.get("tok") for r in rows):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
        for eff in EFF_ORDER:
            pts_t = [(r["step"], r["tok"][eff]) for r in rows if eff in r.get("tok", {})]
            pts_a = [(r["step"], r["eacc"][eff]) for r in rows if eff in r.get("eacc", {})]
            if not pts_t:
                continue
            ax1.plot([p[0] for p in pts_t], ema([p[1] for p in pts_t], 0.06),
                     color=EFF_COLOR[eff], label=f"effort={eff}")
            if pts_a:
                ax2.plot([p[0] for p in pts_a], ema([p[1] for p in pts_a], 0.06),
                         color=EFF_COLOR[eff], label=f"effort={eff}")
        ax1.set_ylabel("generated tokens / rollout (EMA)")
        ax1.set_title("Effort dial: R = r_task − λ·tokens, λ paired with the system message")
        ax1.legend(); ax1.grid(alpha=0.3)
        ax2.set_ylabel("accuracy (EMA)"); ax2.set_xlabel("learner step")
        ax2.set_ylim(-0.05, 1.05); ax2.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "rl_effort.png"), dpi=120); plt.close(fig)
        print("wrote rl_effort.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_root", default="/kaggle/working/checkpoints")
    ap.add_argument("--out", default="/kaggle/working/tiny-inkling/assets")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    plot_pretrain(os.path.join(args.ckpt_root, "pretrain"), args.out)
    plot_rl(os.path.join(args.ckpt_root, "rl_qwen"), args.out)


if __name__ == "__main__":
    main()
