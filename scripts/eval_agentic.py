"""Evaluate agentic tool-use success — base model vs RL-tuned adapter.

Runs a fixed suite of tasks per environment (greedy, no exploration noise) and
reports: final-answer accuracy, tool-use rate, mean task reward, and mean
GENERATED-TOKEN count. Point it at the LoRA adapter dir with --adapter to measure
the *after-RL* policy, or omit it for the *before* baseline. Same tasks/seed for
both so the comparison is apples-to-apples.

The suite is repeated once per --efforts level (same tasks each time), which is
how we measure the effort dial: after effort-conditioned RL, "Effort: low" should
spend far fewer tokens than "Effort: high" AT THE SAME accuracy — and the spread
should be much wider than the instruction alone produces on the base model.
"""
import os, sys, json, argparse, random
sys.path.insert(0, os.path.dirname(__file__))
import torch
from envs import make_env
from rl_agentic import rollout_group


def evaluate(model, tok, env_names, n_per_env, device, efforts=("none",), seed=12345):
    out = {}
    for name in env_names:
        env = make_env(name)
        for eff in efforts:
            rng = random.Random(seed)           # SAME tasks for every (model, effort) pair
            acc = tool = rew = chained = toks = 0.0
            for _ in range(n_per_env):
                task = env.sample_task(rng)
                # greedy single rollout (G=1, temp=0) = the model's best-effort answer
                g = rollout_group(model, tok, env, task, device, G=1, temperature=0.0,
                                  effort=eff)
                info = g[0]["info"]
                acc += bool(info.get("correct")); tool += bool(info.get("used_tool"))
                chained += bool(info.get("chained", False))
                rew += info["task_reward"]      # UNpenalized — the penalty is a training device
                toks += info["n_gen"]
            n = n_per_env
            out[f"{name}/{eff}"] = dict(
                accuracy=round(acc / n, 3), tool_use=round(tool / n, 3),
                task_reward=round(rew / n, 3), chained=round(chained / n, 3),
                mean_gen_tokens=round(toks / n, 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (omit for baseline)")
    ap.add_argument("--envs", default="calc,lookup")
    ap.add_argument("--efforts", default="none,low,medium,high",
                    help="effort levels to sweep; 'none' = no effort line (unconditioned)")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float16).to(args.device)
    tag = "BASE (before RL)"
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).to(args.device)
        tag = f"RL-tuned ({args.adapter})"
    model.eval()

    res = evaluate(model, tok, args.envs.split(","), args.n, args.device,
                   efforts=[e.strip() for e in args.efforts.split(",")])
    print(f"\n=== {tag} — {args.n} tasks/env ===")
    print(json.dumps(res, indent=2))
    if args.out:
        json.dump({"tag": tag, "result": res}, open(args.out, "w"), indent=2)
        print("saved ->", args.out)


if __name__ == "__main__":
    main()
