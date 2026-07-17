"""Synthetic, *verifiable* agentic environments.

"Agentic" here means the model must **take actions** (call tools) over **multiple
turns** and is judged only by whether the **final answer is correct** — a
*verifiable reward*. There are no labels on the reasoning; the environment is the
teacher. This is exactly the setting large-scale agentic RL operates in, just
shrunk to things we can generate infinitely and check exactly.

Transcript protocol (all markup tokens are atomic in our tokenizer):

    <user> ...question... </user>
    <assistant> <think>...</think> <tool>calc(3847*291)</tool> </assistant>
    <result> 1119477 </result>                       <- injected by the ENV
    <assistant> <answer>1119477</answer> </assistant>

The env drives the loop: it reads the model's latest <tool>…</tool>, executes it,
and appends <result>…</result>. Generation resumes until <answer>…</answer> or a
turn budget is hit. `verify()` then returns a scalar reward.

Two environments, deliberately targeting different agentic skills:
  * CalcEnv   — learn to OFFLOAD computation you can't do in-weights to a tool.
  * LookupEnv — learn MULTI-HOP tool use: chain lookups, reason over results.
"""
from __future__ import annotations
import re, random, ast, operator

# ---- a tiny safe arithmetic evaluator for the calc() tool ------------------
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.floordiv, ast.USub: operator.neg, ast.Pow: operator.pow}

def _safe_eval(expr: str):
    """Evaluate a pure-arithmetic expression with no names/calls. Returns int or None."""
    try:
        node = ast.parse(expr, mode="eval").body
    except Exception:
        return None
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError("unsafe")
    try:
        v = ev(node)
        return int(v)
    except Exception:
        return None


TOOL_RE = re.compile(r"<tool>\s*(\w+)\s*\(\s*(.*?)\s*\)\s*</tool>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)


class Environment:
    name = "base"
    def sample_task(self, rng: random.Random) -> dict: ...
    def run_tool(self, task: dict, tool: str, arg: str) -> str: ...
    def verify(self, task: dict, transcript: str) -> dict: ...

    # ---- shared helpers ----
    @staticmethod
    def last_tool_call(text: str):
        m = None
        for m in TOOL_RE.finditer(text):
            pass
        return (m.group(1), m.group(2)) if m else None

    @staticmethod
    def final_answer(text: str):
        m = None
        for m in ANSWER_RE.finditer(text):
            pass
        return m.group(1).strip() if m else None


class CalcEnv(Environment):
    """Multi-digit arithmetic the model can't reliably do in its head → must use calc().

    Reward (in [0,1], shaped so partial competence is rewarded and learning has signal):
        +0.15  emitted a syntactically valid <tool>calc(...)</tool>
        +0.15  the tool's result is the *correct* value (used the tool correctly)
        +0.70  the final <answer> equals the ground truth
        −0.20  answered without ever calling the tool but got it wrong (discourage guessing)
    """
    name = "calc"

    def sample_task(self, rng):
        op = rng.choice(["*", "+", "-"])
        if op == "*":
            a, b = rng.randint(12, 9999), rng.randint(12, 999)
        else:
            a, b = rng.randint(1000, 999999), rng.randint(1000, 999999)
        gt = _safe_eval(f"{a}{op}{b}")
        q = f"What is {a} {op} {b}? Use the calculator tool, then give the answer."
        return {"question": q, "expr": f"{a}{op}{b}", "gt": str(gt)}

    def run_tool(self, task, tool, arg):
        if tool != "calc":
            return f"error: unknown tool '{tool}'"
        v = _safe_eval(arg)
        return "error: bad expression" if v is None else str(v)

    def verify(self, task, transcript):
        r, info = 0.0, {}
        call = self.last_tool_call(transcript)
        used_tool = call is not None and call[0] == "calc"
        tool_correct = False
        if used_tool:
            r += 0.15
            v = _safe_eval(call[1])
            tool_correct = (v is not None and str(v) == task["gt"])
            if tool_correct:
                r += 0.15
        ans = self.final_answer(transcript)
        correct = (ans is not None and ans == task["gt"])
        if correct:
            r += 0.70
        elif not used_tool:
            r -= 0.20   # guessed without the tool and got it wrong
        info.update(used_tool=used_tool, tool_correct=tool_correct, correct=correct)
        return {"reward": max(-0.2, min(1.0, r)), **info}


class LookupEnv(Environment):
    """Multi-hop lookup over a per-task knowledge base → chained tool use.

    Task: 'person -> pet -> city'. To answer "which city does <person>'s pet live
    in?" the model must lookup(person) -> pet, then lookup(pet) -> city. Two hops,
    both via the tool, then answer. Reward mirrors CalcEnv's shaping.
    """
    name = "lookup"
    PEOPLE = ["alice", "bob", "carol", "dave", "erin", "frank", "grace", "heidi"]
    PETS = ["rose", "toby", "milo", "luna", "ziggy", "coco", "pip", "nala"]
    CITIES = ["paris", "tokyo", "cairo", "lima", "oslo", "delhi", "accra", "quito"]

    def sample_task(self, rng):
        person = rng.choice(self.PEOPLE)
        pet = rng.choice(self.PETS)
        city = rng.choice(self.CITIES)
        kb = {person: pet, pet: city}
        # a couple of distractor entries to make blind guessing hard
        for _ in range(2):
            k = rng.choice(self.PEOPLE + self.PETS)
            if k not in kb:
                kb[k] = rng.choice(self.PETS + self.CITIES)
        q = (f"{person.title()} has a pet. Use lookup(name) to find {person.title()}'s pet, "
             f"then lookup(pet) to find which city it lives in, then answer with the city.")
        return {"question": q, "kb": kb, "gt": city}

    def run_tool(self, task, tool, arg):
        if tool != "lookup":
            return f"error: unknown tool '{tool}'"
        return task["kb"].get(arg.strip().lower(), "not found")

    def verify(self, task, transcript):
        r, info = 0.0, {}
        n_tool = len(TOOL_RE.findall(transcript))
        used_tool = n_tool >= 1
        chained = n_tool >= 2                 # two-hop chaining attempted
        if used_tool: r += 0.15
        if chained:   r += 0.15
        ans = self.final_answer(transcript)
        correct = (ans is not None and ans.strip().lower() == task["gt"])
        if correct: r += 0.70
        elif not used_tool: r -= 0.20
        info.update(used_tool=used_tool, chained=chained, correct=correct)
        return {"reward": max(-0.2, min(1.0, r)), **info}


ENVS = {"calc": CalcEnv, "lookup": LookupEnv}


def make_env(name):
    return ENVS[name]()


if __name__ == "__main__":
    # quick self-check
    rng = random.Random(0)
    for name in ENVS:
        env = make_env(name)
        t = env.sample_task(rng)
        print(f"\n[{name}] {t['question']}")
        if name == "calc":
            res = env.run_tool(t, "calc", t["expr"])
            fake = f"<assistant><tool>calc({t['expr']})</tool></assistant><result>{res}</result><assistant><answer>{res}</answer></assistant>"
        else:
            p = list(t["kb"])[0]; pet = t["kb"][p]; city = t["kb"][pet]
            fake = (f"<tool>lookup({p})</tool><result>{pet}</result>"
                    f"<tool>lookup({pet})</tool><result>{city}</result><answer>{city}</answer>")
        print("  gt:", t["gt"], "| verify:", env.verify(t, fake))
