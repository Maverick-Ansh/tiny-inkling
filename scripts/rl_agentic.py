"""
Asynchronous agentic RL — the centerpiece.

We take a small real foundation model (Qwen2.5-0.5B-Instruct), attach LoRA
adapters, and train it with **GRPO** to solve *verifiable* tool-use tasks, using
an **asynchronous stale-actor / learner** setup with an **IcePop** off-policy
correction. Read grpo.py first for the objective; read envs.py for the tasks.

────────────────────────────────────────────────────────────────────────────
The async architecture (why it looks like "large-scale asynchronous RL")
────────────────────────────────────────────────────────────────────────────
Two model replicas, one per GPU:

    ┌─────────────── GPU 1 ───────────────┐        ┌─────────── GPU 0 ───────────┐
    │  ACTOR (eval, LoRA snapshot)        │  queue │  LEARNER (train, LoRA)      │
    │  • sample task from an env          │ ─────► │  • pop a group              │
    │  • roll out a GROUP of G answers    │        │  • recompute logπ_θ         │
    │    (multi-turn: generate→tool→...)  │        │  • GRPO + IcePop update     │
    │  • score each with env.verify()     │        │  • every K steps:           │
    │  • compute behavior logπ_behavior   │ ◄───── │      sync LoRA → actor      │
    └─────────────────────────────────────┘  sync  └─────────────────────────────┘

The actor runs in a **background thread** and keeps generating with a snapshot of
the weights that is refreshed only every `sync_every` learner steps. So by the
time the learner consumes a rollout, the policy has already moved — the data is
**off-policy / stale**. That is the defining feature of asynchronous RL (it's how
you keep expensive generation and cheap updates both busy at scale), and it's
exactly what the importance ratio + IcePop mask in grpo.py exist to correct.

We log the *staleness* (updates since the actor's snapshot) and the *IcePop mask
fraction* so you can watch the correction working.
"""
import os, sys, time, json, math, argparse, threading, queue, random, copy
sys.path.insert(0, os.path.dirname(__file__))
import torch
import torch.nn.functional as F

from envs import make_env
from grpo import group_advantages, grpo_loss, gather_token_logprobs
from common import append_jsonl


# ───────────────────────── prompt / rollout helpers ─────────────────────────
SYS_TMPL = (
    "You are a precise tool-using assistant. Think briefly, then act.\n"
    "To call a tool, output exactly <tool>name(argument)</tool> and stop; you will "
    "then be shown <result>...</result>. When you know the answer, output it as "
    "<answer>...</answer> and nothing else.\n"
    "Available tool: {tool_desc}\n"
    "Always use the tool before answering. Follow this example exactly:\n{example}"
)
TOOL_DESC = {
    "calc": "calc(expression) — evaluates integer arithmetic, e.g. <tool>calc(12*34)</tool>.",
    "lookup": "lookup(name) — returns the value stored for a lowercase name, e.g. <tool>lookup(alice)</tool>.",
}
# One-shot demonstrations of the EXACT protocol. A 0.5B model won't reliably invent
# the <tool>/<answer> format; priming it in-context lets RL optimise *correctness*
# rather than spend all its signal discovering the syntax.
ONESHOT = {
    "calc": (
        "Q: What is 12 * 34?\n"
        "<tool>calc(12*34)</tool>\n<result>408</result>\n<answer>408</answer>"
    ),
    "lookup": (
        "Q: Zoe has a pet. Find Zoe's pet, then its city.\n"
        "<tool>lookup(zoe)</tool>\n<result>fig</result>\n"
        "<tool>lookup(fig)</tool>\n<result>rome</result>\n<answer>rome</answer>"
    ),
}


def build_prompt(tok, env_name, question):
    msgs = [{"role": "system", "content": SYS_TMPL.format(
                tool_desc=TOOL_DESC[env_name], example=ONESHOT[env_name])},
            {"role": "user", "content": question}]
    # tokenize=False -> formatted string; then encode to a flat list[int] ourselves
    # (avoids version-dependent return types from apply_chat_template).
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return tok.encode(text, add_special_tokens=False)


@torch.no_grad()
def _batch_generate(model, tok, id_lists, device, max_new, stop_strings, temperature):
    """Left-pad `id_lists`, generate, return the list of NEW token-id lists
    (per sequence, with trailing pad and anything after the stop string removed)."""
    pad_id = tok.pad_token_id
    maxlen = max(len(x) for x in id_lists)
    input_ids, attn = [], []
    for x in id_lists:
        p = maxlen - len(x)
        input_ids.append([pad_id] * p + x)     # LEFT pad (so generation continues at the right edge)
        attn.append([0] * p + [1] * len(x))
    input_ids = torch.tensor(input_ids, device=device)
    attn = torch.tensor(attn, device=device)
    out = model.generate(
        input_ids=input_ids, attention_mask=attn,
        max_new_tokens=max_new, do_sample=temperature > 0, temperature=max(temperature, 1e-5),
        top_p=0.95, pad_token_id=pad_id, tokenizer=tok, stop_strings=stop_strings,
    )
    gen = out[:, maxlen:]                        # only the newly generated part
    res = []
    for row in gen.tolist():
        # strip trailing pads (finished sequences are pad-filled after their stop string)
        while row and row[-1] == pad_id:
            row.pop()
        res.append(row)
    return res


def rollout_group(model, tok, env, task, device, G, max_turns=4, max_new=80, temperature=1.0):
    """Roll out G independent completions for one task (a GRPO group).

    Returns a list of G dicts: {ids, resp_mask, reward, info}. `resp_mask` is 1 on
    tokens the MODEL generated (assistant turns) and 0 on the prompt and on the
    <result> text the environment injected — we only train on the model's tokens.
    """
    prompt = build_prompt(tok, env.name, task["question"])
    seqs = [list(prompt) for _ in range(G)]
    mask = [[0] * len(prompt) for _ in range(G)]     # prompt tokens -> 0
    done = [False] * G

    for _turn in range(max_turns):
        active = [i for i in range(G) if not done[i]]
        if not active:
            break
        new = _batch_generate(model, tok, [seqs[i] for i in active], device,
                              max_new, ["</tool>", "</answer>"], temperature)
        for j, i in enumerate(active):
            gen_ids = new[j]
            if not gen_ids:
                done[i] = True
                continue
            seqs[i] += gen_ids
            mask[i] += [1] * len(gen_ids)            # model-generated -> 1
            tail = tok.decode(gen_ids)               # THIS turn's text only
            if "</answer>" in tail:
                done[i] = True
            elif "</tool>" in tail:
                # a NEW tool call was emitted THIS turn — parse from `tail`, not the
                # whole transcript, so we never re-execute an already-answered call.
                call = env.last_tool_call(tail)
                if call is not None:
                    result = env.run_tool(task, call[0], call[1])
                    inj = tok.encode(f"\n<result>{result}</result>\n", add_special_tokens=False)
                    seqs[i] += inj
                    mask[i] += [0] * len(inj)         # injected by env -> 0
                else:
                    done[i] = True
            else:
                done[i] = True                        # turn ended (eos/maxlen), no tool/answer

    out = []
    for i in range(G):
        v = env.verify(task, tok.decode(seqs[i]))
        out.append({"ids": seqs[i], "resp_mask": mask[i], "reward": v["reward"], "info": v})
    return out


# ───────────────────────── logprob / batching for the learner ─────────────────────────
def pad_stack(id_lists, mask_lists, pad_id, device):
    T = max(len(x) for x in id_lists)
    ids, msk, attn = [], [], []
    for x, m in zip(id_lists, mask_lists):
        p = T - len(x)
        ids.append(x + [pad_id] * p)
        msk.append(m + [0] * p)                       # padding -> not a response token
        attn.append([1] * len(x) + [0] * p)
    return (torch.tensor(ids, device=device), torch.tensor(msk, device=device, dtype=torch.float),
            torch.tensor(attn, device=device))


def seq_logprobs(model, ids, attn, use_adapter=True):
    """Per-token logπ for the realised next token, shape (N, T-1). If use_adapter
    is False and the model is a PEFT model, computes the REFERENCE (base) logprobs."""
    ctx = model.disable_adapter() if (not use_adapter and hasattr(model, "disable_adapter")) else _null()
    with ctx:
        logits = model(input_ids=ids, attention_mask=attn).logits
    return gather_token_logprobs(logits, ids)         # (N, T-1)


class _null:
    def __enter__(self): return self
    def __exit__(self, *a): return False


# ───────────────────────── the async actor thread ─────────────────────────
class Actor(threading.Thread):
    def __init__(self, model, tok, envs, device, G, out_q, stop_flag, cfg):
        super().__init__(daemon=True)
        self.model, self.tok, self.envs = model, tok, envs
        self.device, self.G, self.q = device, G, out_q
        self.stop_flag, self.cfg = stop_flag, cfg
        self.snapshot_version = 0
        self.rng = random.Random(0)
        self.lock = threading.Lock()

    def run(self):
        self.model.eval()
        while not self.stop_flag.is_set():
            env = self.envs[self.rng.randrange(len(self.envs))]
            task = env.sample_task(self.rng)
            with self.lock:                            # don't generate while syncing weights
                group = rollout_group(self.model, self.tok, env, task, self.device,
                                      self.G, temperature=self.cfg.temperature)
                # behavior logprobs under the ACTOR snapshot (this is logπ_behavior)
                ids, msk = [g["ids"] for g in group], [g["resp_mask"] for g in group]
                t_ids, t_msk, t_attn = pad_stack(ids, msk, self.tok.pad_token_id, self.device)
                with torch.no_grad():
                    b_logp = seq_logprobs(self.model, t_ids, t_attn).cpu()
                ver = self.snapshot_version
            item = dict(env=env.name, ids=ids, resp_mask=msk, b_logp=b_logp,
                        rewards=[g["reward"] for g in group],
                        infos=[g["info"] for g in group], version=ver)
            try:
                self.q.put(item, timeout=5)
            except queue.Full:
                pass                                    # learner is behind; drop (backpressure)

    def sync_from(self, learner_state, version):
        """Load fresh LoRA weights from the learner (called by the main thread).

        MUST use set_peft_model_state_dict — the keys from get_peft_model_state_dict
        omit the adapter name ('.default'), so a plain load_state_dict(strict=False)
        would match nothing and the actor would silently never update."""
        from peft import set_peft_model_state_dict
        with self.lock:
            set_peft_model_state_dict(self.model, learner_state)
            self.snapshot_version = version


# ───────────────────────── main training loop (learner) ─────────────────────────
def lora_state_on(model, device):
    from peft import get_peft_model_state_dict
    sd = get_peft_model_state_dict(model)
    return {k: v.detach().to(device) for k, v in sd.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--envs", default="calc,lookup")
    ap.add_argument("--group_size", type=int, default=8)     # G rollouts per task
    ap.add_argument("--groups_per_step", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--clip_eps", type=float, default=0.2)
    ap.add_argument("--icepop_c", type=float, default=2.0)   # IcePop trust band [1/c, c]
    ap.add_argument("--kl_beta", type=float, default=0.02)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--sync_every", type=int, default=4)     # learner steps between actor syncs
    ap.add_argument("--total_steps", type=int, default=600)
    ap.add_argument("--max_minutes", type=float, default=110.0)
    ap.add_argument("--out", default="/kaggle/working/checkpoints/rl_qwen")
    ap.add_argument("--hf_repo", default="AnshVivek/tiny-inkling-rl-qwen")
    ap.add_argument("--sync_mode", default="async", choices=["async", "sync"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])

    # LEARNER policy on cuda:0 in fp32 — training LoRA in fp16 without loss-scaling
    # underflows over a long run; fp32 is stable and a 0.5B base fits easily.
    policy = get_peft_model(
        AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float32).to("cuda:0"), lcfg)
    policy.print_trainable_parameters()
    # ACTOR on cuda:1 in fp16 — generation only, so speed matters and stability doesn't.
    actor = get_peft_model(
        AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float16).to("cuda:1"), lcfg)
    # initial sync: copy policy's (fp32) weights into the actor (cast to fp16 on copy)
    from peft import set_peft_model_state_dict
    set_peft_model_state_dict(actor, lora_state_on(policy, "cuda:1"))

    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad],
                            lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    envs = [make_env(n) for n in args.envs.split(",")]

    stop_flag = threading.Event()
    q = queue.Queue(maxsize=8)
    actor_thread = None
    if args.sync_mode == "async":
        actor_thread = Actor(actor, tok, envs, "cuda:1", args.group_size, q, stop_flag, args)
        actor_thread.start()

    log_path = os.path.join(args.out, "rl_log.jsonl")
    t_start = time.time()
    rng = random.Random(1)
    ema = {}

    def get_group_async():
        return q.get(timeout=120)

    def get_group_sync():
        env = envs[rng.randrange(len(envs))]
        task = env.sample_task(rng)
        actor.eval()
        group = rollout_group(actor, tok, env, task, "cuda:1", args.group_size, temperature=args.temperature)
        ids, msk = [g["ids"] for g in group], [g["resp_mask"] for g in group]
        t_ids, t_msk, t_attn = pad_stack(ids, msk, tok.pad_token_id, "cuda:1")
        with torch.no_grad():
            b_logp = seq_logprobs(actor, t_ids, t_attn).cpu()
        return dict(env=env.name, ids=ids, resp_mask=msk, b_logp=b_logp,
                    rewards=[g["reward"] for g in group],
                    infos=[g["info"] for g in group], version=0)

    for step in range(args.total_steps):
        # ---- pull `groups_per_step` groups (from the async queue or generate now) ----
        groups = []
        for _ in range(args.groups_per_step):
            groups.append(get_group_async() if args.sync_mode == "async" else get_group_sync())

        # flatten to a batch of sequences, remembering group boundaries for advantages
        all_ids, all_mask, all_rew, b_logp_rows, versions = [], [], [], [], []
        for g in groups:
            all_ids += g["ids"]; all_mask += g["resp_mask"]; all_rew += g["rewards"]
            b_logp_rows.append(g["b_logp"]); versions.append(g["version"])
        rewards = torch.tensor(all_rew)
        adv = group_advantages(rewards, args.group_size).to("cuda:0")

        # pad the whole batch on the learner GPU
        ids, resp_mask, attn = pad_stack(all_ids, all_mask, tok.pad_token_id, "cuda:0")
        # behavior logprobs: pad each group's rows to the batch width, then concat
        Tm1 = ids.shape[1] - 1
        b_logp = torch.zeros(len(all_ids), Tm1)
        row = 0
        for br in b_logp_rows:
            n, t = br.shape
            b_logp[row:row + n, :t] = br
            row += n
        b_logp = b_logp.to("cuda:0")

        # ---- learner forward: current-policy and reference logprobs ----
        policy.train()
        n_logp = seq_logprobs(policy, ids, attn, use_adapter=True)          # logπ_θ  (grad)
        with torch.no_grad():
            r_logp = seq_logprobs(policy, ids, attn, use_adapter=False)     # logπ_ref
        rmask = resp_mask[:, 1:]                                            # align with (T-1) logprobs

        loss, st = grpo_loss(n_logp, b_logp, adv, rmask, clip_eps=args.clip_eps,
                             icepop_c=args.icepop_c, logp_ref=r_logp, kl_beta=args.kl_beta)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 1.0)
        opt.step()

        # ---- periodic actor sync (defines the staleness) ----
        cur_version = step + 1
        staleness = cur_version - (sum(versions) / len(versions))
        if (step + 1) % args.sync_every == 0:
            new_state = lora_state_on(policy, "cuda:1")
            if args.sync_mode == "async":
                actor_thread.sync_from(new_state, cur_version)
            else:
                from peft import set_peft_model_state_dict
                set_peft_model_state_dict(actor, new_state)

        # ---- metrics ----
        infos = [i for g in groups for i in g["infos"]]
        acc = sum(bool(i.get("correct")) for i in infos) / len(infos)
        tool = sum(bool(i.get("used_tool")) for i in infos) / len(infos)
        rec = dict(step=step, loss=round(float(loss), 4), reward=round(float(rewards.mean()), 3),
                   acc=round(acc, 3), tool_use=round(tool, 3), staleness=round(float(staleness), 2),
                   icepop=round(st["icepop_masked_frac"], 3), kl=round(st["kl"], 4),
                   clip=round(st["clip_frac"], 3), qsize=q.qsize() if args.sync_mode == "async" else 0)
        for k in ("reward", "acc", "tool_use"):
            ema[k] = rec[k] if k not in ema else 0.9 * ema[k] + 0.1 * rec[k]
        if step % 5 == 0:
            print({**rec, "ema_reward": round(ema["reward"], 3), "ema_acc": round(ema["acc"], 3)}, flush=True)
        append_jsonl(log_path, rec)

        # ---- checkpoint (LoRA adapter) + HF push ----
        if (step + 1) % 100 == 0:
            policy.save_pretrained(args.out)
            try:
                from huggingface_hub import HfApi
                HfApi(token=os.environ.get("HF_TOKEN")).upload_folder(
                    folder_path=args.out, repo_id=args.hf_repo, repo_type="model")
            except Exception as e:
                print("hf push warn:", e)

        if (time.time() - t_start) > args.max_minutes * 60:
            print("[time budget] stopping", flush=True)
            break

    stop_flag.set()
    policy.save_pretrained(args.out)
    print("DONE. final ema:", {k: round(v, 3) for k, v in ema.items()}, flush=True)


if __name__ == "__main__":
    main()
