"""Supervised fine-tuning of pretrained Inkling-mini on synthetic tool-use traces.

Purpose (Track A): give our from-scratch model the *format* of agentic tool use —
<user>…</user>, <tool>…</tool>, injected <result>…</result>, <answer>…</answer> —
so the later RL loop has a sane starting policy. We only compute loss on the
**assistant's own tokens** (the tool call and the final answer); the user turn and
the environment's <result> are masked with -100 so the model isn't trained to
predict text it doesn't control. This "loss masking on non-assistant tokens" is
the single most important detail in SFT-for-agents.

A 70M model pretrained on TinyStories will only ever be *toy* at this — the point
is that the same pipeline (SFT → RL) that works on Qwen (Track B) runs on our own
architecture end-to-end.
"""
import os, sys, json, time, random, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch
from tokenizers import Tokenizer
from tiny_inkling import InklingForCausalLM, InklingConfig, build_param_groups, MuonWithAuxAdam
from common import CheckpointManager, append_jsonl, cosine_lr, set_seed
sys.path.insert(0, os.path.dirname(__file__))
from envs import make_env


def gold_segments(env, task):
    """Return a list of (text, trainable?) segments forming a *correct* trace."""
    q = task["question"]
    if env.name == "calc":
        r = env.run_tool(task, "calc", task["expr"])
        return [
            (f"<user>{q}</user>", False),
            ("<assistant>", False),
            (f"<tool>calc({task['expr']})</tool>", True),
            ("</assistant>", False),
            (f"<result>{r}</result>", False),
            ("<assistant>", False),
            (f"<answer>{task['gt']}</answer>", True),
            ("</assistant><|endoftext|>", False),
        ]
    else:  # lookup — two hops
        person = next(iter(task["kb"]))
        pet = task["kb"][person]
        city = task["kb"][pet]
        return [
            (f"<user>{q}</user>", False),
            ("<assistant>", False), (f"<tool>lookup({person})</tool>", True), ("</assistant>", False),
            (f"<result>{pet}</result>", False),
            ("<assistant>", False), (f"<tool>lookup({pet})</tool>", True), ("</assistant>", False),
            (f"<result>{city}</result>", False),
            ("<assistant>", False), (f"<answer>{city}</answer>", True),
            ("</assistant><|endoftext|>", False),
        ]


def build_example(tok, env, task, max_len):
    ids, labels = [], []
    for text, trainable in gold_segments(env, task):
        seg = tok.encode(text).ids
        ids += seg
        labels += (seg if trainable else [-100] * len(seg))   # -100 == ignore in CE
    ids, labels = ids[:max_len], labels[:max_len]
    return ids, labels


def collate(batch, pad_id, max_len):
    T = min(max_len, max(len(x[0]) for x in batch))
    ids = np.full((len(batch), T), pad_id, np.int64)
    lab = np.full((len(batch), T), -100, np.int64)
    for i, (x, y) in enumerate(batch):
        n = min(len(x), T)
        ids[i, :n] = x[:n]; lab[i, :n] = y[:n]
    return torch.from_numpy(ids), torch.from_numpy(lab)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", default="/kaggle/working/tok/tiny8k.json")
    ap.add_argument("--init_repo", default="AnshVivek/tiny-inkling-pretrain")
    ap.add_argument("--init_local", default="/kaggle/working/checkpoints/pretrain/latest.pt")
    ap.add_argument("--out", default="/kaggle/working/checkpoints/sft")
    ap.add_argument("--hf_repo", default="AnshVivek/tiny-inkling-sft")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max_len", type=int, default=256)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    set_seed(0)
    device = "cuda:0"

    tok = Tokenizer.from_file(args.tok)
    pad_id = tok.token_to_id("<|pad|>")

    # load pretrained weights (local first, then Hub)
    cm = CheckpointManager(os.path.dirname(args.init_local), hf_repo=args.init_repo)
    state = cm.load_latest(name=os.path.basename(args.init_local))
    cfg = InklingConfig(**{k: v for k, v in state["cfg"].items()}) if state and "cfg" in state \
        else InklingConfig(vocab_size=tok.get_vocab_size(), max_seq_len=args.max_len)
    model = InklingForCausalLM(cfg).to(device)
    if state is not None:
        model.load_state_dict(state["model"]); print("loaded pretrained weights")
    else:
        print("WARNING: no pretrained ckpt found — SFT from random init (demo only)")

    opt = MuonWithAuxAdam(build_param_groups(model, lr=args.lr))
    scaler = torch.cuda.amp.GradScaler()
    envs = [make_env("calc"), make_env("lookup")]
    rng = random.Random(0)
    out_ckpt = CheckpointManager(args.out, hf_repo=args.hf_repo)

    model.train()
    log = os.path.join(args.out, "sft_log.jsonl")
    for step in range(args.steps):
        batch = []
        for _ in range(args.bs):
            env = envs[rng.randrange(2)]
            batch.append(build_example(tok, env, env.sample_task(rng), args.max_len))
        ids, lab = collate(batch, pad_id, args.max_len)
        ids, lab = ids.to(device), lab.to(device)
        lr = cosine_lr(step, warmup=50, total=args.steps, base_lr=args.lr)
        for g in opt.param_groups:
            g["lr"] = lr if g["use_muon"] else lr * 0.1
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(ids, labels=lab)["loss"]
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        if step % 50 == 0:
            rec = dict(step=step, loss=round(float(loss), 4), lr=round(lr, 6))
            print(rec, flush=True); append_jsonl(log, rec)
    out_ckpt.save(dict(model=model.state_dict(), cfg=cfg.__dict__, step=args.steps), name="latest.pt")
    print("SFT done ->", args.out, flush=True)


if __name__ == "__main__":
    main()
