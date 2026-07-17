"""Pretrain Inkling-mini on TinyStories over 2×T4 with DDP + fp16.

Design points that matter for *this* hardware / setting:

  * fp16, not bf16. T4 (Turing) has no bf16 tensor cores, so bf16 falls back to
    slow emulation. We use fp16 autocast + a GradScaler (loss scaling handles the
    narrow fp16 exponent range). The RMSNorm/softmax reductions are done in fp32
    inside the modules for stability.

  * DDP with find_unused_parameters=True. In an MoE step only the *selected*
    experts receive gradients; the rest are "unused" in that microbatch, which
    plain DDP would treat as an error. The flag tells DDP to expect it.

  * Time-boxed + checkpointed. Kaggle caps session length; we save a rolling
    checkpoint to the HF Hub every few minutes and on exit, and resume from it on
    the next session. The optimiser, scaler, step, and RNG are all restored.

Launch (2 GPUs):
    torchrun --nproc_per_node=2 scripts/pretrain.py --max_minutes 110
"""
import os, sys, json, time, argparse, math
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tiny_inkling import InklingForCausalLM, InklingConfig, build_param_groups, MuonWithAuxAdam
from common import set_seed, cosine_lr, BinDataset, CheckpointManager, append_jsonl


def is_dist():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/kaggle/working/data")
    ap.add_argument("--out", default="/kaggle/working/checkpoints/pretrain")
    ap.add_argument("--hf_repo", default="AnshVivek/tiny-inkling-pretrain")
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--micro_bs", type=int, default=16)     # per-GPU microbatch
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-3)       # Muon lr
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--total_steps", type=int, default=12000)
    ap.add_argument("--wd", type=float, default=0.1)        # BASE lambda; effective = wd*lr^2
    ap.add_argument("--ckpt_every_min", type=float, default=6.0)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--max_minutes", type=float, default=110.0)
    args = ap.parse_args()

    # ---- distributed setup ----
    if is_dist():
        dist.init_process_group("nccl")
        rank = dist.get_rank(); world = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank, world, local_rank = 0, 1, 0
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(local_rank)
    is_main = (rank == 0)
    set_seed(1234 + rank)

    meta = json.load(open(os.path.join(args.data, "meta.json")))
    cfg = InklingConfig(vocab_size=meta["vocab_size"], max_seq_len=args.seq_len)
    if is_main:
        os.makedirs(args.out, exist_ok=True)
        cfg.to_json(os.path.join(args.out, "config.json"))

    model = InklingForCausalLM(cfg).to(device)
    if is_main:
        print(f"params total={model.num_params():,} active/tok≈{model.num_params(only_active=True):,}", flush=True)

    opt = MuonWithAuxAdam(build_param_groups(model, lr=args.lr, weight_decay=args.wd))
    scaler = torch.cuda.amp.GradScaler()

    ckpt = CheckpointManager(args.out, hf_repo=args.hf_repo, is_main=is_main)
    start_step = 0
    state = ckpt.load_latest()
    if state is not None:
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        scaler.load_state_dict(state["scaler"])
        start_step = state["step"]
        if is_main:
            print(f"resumed at step {start_step}", flush=True)

    if is_dist():
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True,
                    broadcast_buffers=True)
    raw = model.module if is_dist() else model

    train = BinDataset(os.path.join(args.data, "train.bin"), args.seq_len)
    val = BinDataset(os.path.join(args.data, "val.bin"), args.seq_len)

    @torch.no_grad()
    def evaluate(n=20):
        raw.eval()
        tot = 0.0
        for _ in range(n):
            x, y = val.batch(args.micro_bs, device)
            with torch.autocast("cuda", dtype=torch.float16):
                tot += raw(x, labels=y)["loss"].item()
        raw.train()
        return tot / n

    model.train()
    t0 = time.time(); last_ckpt = time.time()
    log_path = os.path.join(args.out, "train_log.jsonl")

    for step in range(start_step, args.total_steps):
        lr = cosine_lr(step, warmup=args.warmup, total=args.total_steps, base_lr=args.lr)
        for g in opt.param_groups:
            g["lr"] = lr if g["use_muon"] else lr * 0.1

        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for micro in range(args.grad_accum):
            x, y = train.batch(args.micro_bs, device)
            # DDP grad sync only on the last micro-step (perf)
            sync_ctx = model.no_sync() if (is_dist() and micro < args.grad_accum - 1) else _null()
            with sync_ctx:
                with torch.autocast("cuda", dtype=torch.float16):
                    out = raw(x, labels=y) if not is_dist() else model(x, labels=y)
                    loss = out["loss"] / args.grad_accum
                scaler.scale(loss).backward()
            loss_acc += loss.item()

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        if is_main and step % args.log_every == 0:
            # tokens processed over the whole interval since the last log (log_every steps)
            interval_tokens = args.micro_bs * args.seq_len * args.grad_accum * world * args.log_every
            tok_per_s = interval_tokens / (time.time() - t0 + 1e-9)
            stats = raw.moe_stats()
            rec = dict(step=step, loss=round(loss_acc, 4), lr=round(lr, 6),
                       tok_s=int(tok_per_s), **{k: round(v, 3) for k, v in stats.items()})
            print(rec, flush=True)
            append_jsonl(log_path, rec)
            t0 = time.time()

        # ---- periodic checkpoint (rank 0) ----
        if is_main and (time.time() - last_ckpt) > args.ckpt_every_min * 60:
            vloss = evaluate()
            ckpt.save(dict(model=raw.state_dict(), opt=opt.state_dict(),
                           scaler=scaler.state_dict(), step=step + 1, val_loss=vloss,
                           cfg=cfg.__dict__))
            append_jsonl(log_path, dict(step=step, val_loss=round(vloss, 4), event="ckpt"))
            print(f"[ckpt] step={step} val_loss={vloss:.4f}", flush=True)
            last_ckpt = time.time()

        # ---- session time budget: stop cleanly so the next session resumes ----
        if (time.time() - _START) > args.max_minutes * 60:
            if is_main:
                vloss = evaluate()
                ckpt.save(dict(model=raw.state_dict(), opt=opt.state_dict(),
                               scaler=scaler.state_dict(), step=step + 1, val_loss=vloss,
                               cfg=cfg.__dict__))
                print(f"[time-budget hit] stopping at step {step}, saved. val_loss={vloss:.4f}", flush=True)
            break

    if is_main:
        vloss = evaluate()
        ckpt.save(dict(model=raw.state_dict(), opt=opt.state_dict(),
                       scaler=scaler.state_dict(), step=args.total_steps, val_loss=vloss,
                       cfg=cfg.__dict__), name="final.pt")
        print(f"DONE final val_loss={vloss:.4f}", flush=True)
    if is_dist():
        dist.destroy_process_group()


class _null:
    def __enter__(self): return self
    def __exit__(self, *a): return False


_START = time.time()
if __name__ == "__main__":
    main()
