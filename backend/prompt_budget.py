import os
import math

def approx_tokens(s: str) -> int:
    # conservative 1.3x word count
    return max(1, int(len(s.split()) * 1.3))

def plan_budget(user_prompt: str, instructions: str, model_max: int, frac_ans: float, frac_instr: float):
    reserve_ans = max(128, int(model_max * frac_ans))
    instr_tokens = approx_tokens(instructions) if instructions else 0
    prompt_tokens = approx_tokens(user_prompt)
    context_budget = max(0, model_max - reserve_ans - instr_tokens - prompt_tokens)
    return dict(
        reserve_ans=reserve_ans,
        instr_tokens=instr_tokens,
        prompt_tokens=prompt_tokens,
        context_budget=context_budget
    )

def pack_context(chunks: list, budget: int, sep: str = "\n\n—\n\n"):
    chosen = []
    used = 0
    for c in chunks:
        t = c.get("attached_context") or c["text"]
        cost = approx_tokens(t) + approx_tokens(sep)
        if used + cost > budget:
            break
        chosen.append(t)
        used += cost
    return sep.join(chosen), used