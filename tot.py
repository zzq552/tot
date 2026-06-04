"""
Tree of Thoughts (ToT) — 24-Point Game Solver
==============================================
Reference: "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"
           Yao et al., NeurIPS 2023

Usage:
    python tot.py                    # default: propose + value + greedy, 1 problem
    python tot.py --naive            # single-pass (no search)
    python tot.py --start 0 --end 3  # solve problems 0,1,2
    python tot.py --backend gpt-4o   # use a different model

Dependencies: openai>=1.0, backoff>=2.0, sympy>=1.12, numpy>=1.24
"""

import itertools
import json
import os
import re
import sys
from argparse import ArgumentParser
from functools import partial

import numpy as np
import sympy
from openai import OpenAI

# ============================================================================
# Configuration — edit these or set environment variables
# ============================================================================

API_KEY = os.environ.get("OPENAI_API_KEY", "your-api-key-here")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
completion_tokens = 0
prompt_tokens = 0

# ============================================================================
# Few-shot prompts for the 24-point game
# ============================================================================

STANDARD_PROMPT = """Use numbers and basic arithmetic operations (+ - * /) to obtain 24.
Input: 4 4 6 8
Answer: (4 + 8) * (6 - 4) = 24
Input: 2 9 10 12
Answer: 2 * 12 * (10 - 9) = 24
Input: 4 9 10 13
Answer: (13 - 9) * (10 - 4) = 24
Input: 1 4 8 8
Answer: (8 / 4 + 1) * 8 = 24
Input: 5 5 5 9
Answer: 5 + 5 + 5 + 9 = 24
Input: {input}
"""

COT_PROMPT = """Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Each step, you are only allowed to choose two of the remaining numbers to obtain a new number.
Input: 4 4 6 8
Steps:
4 + 8 = 12 (left: 4 6 12)
6 - 4 = 2 (left: 2 12)
2 * 12 = 24 (left: 24)
Answer: (6 - 4) * (4 + 8) = 24
Input: 2 9 10 12
Steps:
12 * 2 = 24 (left: 9 10 24)
10 - 9 = 1 (left: 1 24)
24 * 1 = 24 (left: 24)
Answer: (12 * 2) * (10 - 9) = 24
Input: 4 9 10 13
Steps:
13 - 10 = 3 (left: 3 4 9)
9 - 3 = 6 (left: 4 6)
4 * 6 = 24 (left: 24)
Answer: 4 * (9 - (13 - 10)) = 24
Input: 1 4 8 8
Steps:
8 / 4 = 2 (left: 1 2 8)
1 + 2 = 3 (left: 3 8)
3 * 8 = 24 (left: 24)
Answer: (1 + 8 / 4) * 8 = 24
Input: 5 5 5 9
Steps:
5 + 5 = 10 (left: 5 9 10)
10 + 5 = 15 (left: 9 15)
15 + 9 = 24 (left: 24)
Answer: ((5 + 5) + 5) + 9 = 24
Input: {input}
"""

PROPOSE_PROMPT = """Input: 2 8 8 14
Possible next steps:
2 + 8 = 10 (left: 8 10 14)
8 / 2 = 4 (left: 4 8 14)
14 + 2 = 16 (left: 8 8 16)
2 * 8 = 16 (left: 8 14 16)
8 - 2 = 6 (left: 6 8 14)
14 - 8 = 6 (left: 2 6 8)
14 /  2 = 7 (left: 7 8 8)
14 - 2 = 12 (left: 8 8 12)
Input: {input}
Possible next steps:
"""

VALUE_PROMPT = """Evaluate if given numbers can reach 24 (sure/likely/impossible)
10 14
10 + 14 = 24
sure
11 12
11 + 12 = 23
12 - 11 = 1
11 * 12 = 132
11 / 12 = 0.91
impossible
4 4 10
4 + 4 + 10 = 8 + 10 = 18
4 * 10 - 4 = 40 - 4 = 36
(10 - 4) * 4 = 6 * 4 = 24
sure
4 9 11
9 + 11 + 4 = 20 + 4 = 24
sure
5 7 8
5 + 7 + 8 = 12 + 8 = 20
(8 - 5) * 7 = 3 * 7 = 21
I cannot obtain 24 now, but numbers are within a reasonable range
likely
5 6 6
5 + 6 + 6 = 17
(6 - 5) * 6 = 1 * 6 = 6
I cannot obtain 24 now, but numbers are within a reasonable range
likely
10 10 11
10 + 10 + 11 = 31
(11 - 10) * 10 = 10
10 10 10 are all too big
impossible
1 3 3
1 * 3 * 3 = 9
(1 + 3) * 3 = 12
1 3 3 are all too small
impossible
{input}
"""

VALUE_LAST_STEP_PROMPT = """Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Given an input and an answer, give a judgement (sure/impossible) if the answer is correct, i.e. it uses each input exactly once and no other numbers, and reach 24.
Input: 4 4 6 8
Answer: (4 + 8) * (6 - 4) = 24
Judge:
sure
Input: 2 9 10 12
Answer: 2 * 12 * (10 - 9) = 24
Judge:
sure
Input: 4 9 10 13
Answer: (13 - 9) * (10 - 4) = 24
Judge:
sure
Input: 4 4 6 8
Answer: (4 + 8) * (6 - 4) + 1 = 25
Judge:
impossible
Input: 2 9 10 12
Answer: 2 * (12 - 10) = 24
Judge:
impossible
Input: 4 9 10 13
Answer: (13 - 4) * (10 - 9) = 24
Judge:
impossible
Input: {input}
Answer: {answer}
Judge:"""

# ============================================================================
# LLM wrapper
# ============================================================================

import backoff


@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def _completions_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)


def gpt(prompt, model=MODEL, temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    """Send a prompt and return *n* response strings."""
    global completion_tokens, prompt_tokens
    outputs = []
    for _ in range(n):
        res = _completions_with_backoff(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
        )
        outputs.extend([c.message.content for c in res.choices])
        completion_tokens += res.usage.completion_tokens
        prompt_tokens += res.usage.prompt_tokens
    return outputs


def gpt_usage():
    global completion_tokens, prompt_tokens
    cost = (completion_tokens + prompt_tokens) / 1000 * 0.005
    return {
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "cost": cost,
    }


# ============================================================================
# Task: 24-Point Game
# ============================================================================

class Game24Task:
    """Represents the 24-point game as a ToT task."""

    def __init__(self):
        # A small built-in test set (from the original 4nums.com benchmark)
        self.data = [
            "1 1 4 6",
            "1 1 11 11",
            "1 1 3 8",
            "1 1 1 8",
            "1 1 5 5",
            "1 1 1 1",
            "1 1 2 7",
            "4 5 6 10",    # (4*5)+(10-6)
            "1 2 4 7",     # (7-1+2)*4
            "2 5 7 9",     # 5*7-2-9
        ]
        self.value_cache = {}
        self.steps = 4
        self.stops = ["\n"] * 4

    def __len__(self):
        return len(self.data)

    def get_input(self, idx: int) -> str:
        return self.data[idx]

    def test_output(self, idx: int, output: str) -> dict:
        expr = (
            output.strip()
            .split("\n")[-1]
            .lower()
            .replace("answer: ", "")
            .split("=")[0]
        )
        numbers = re.findall(r"\d+", expr)
        problem_numbers = re.findall(r"\d+", self.data[idx])
        if sorted(numbers) != sorted(problem_numbers):
            return {"r": 0}
        try:
            return {"r": int(sympy.simplify(expr) == 24)}
        except Exception:
            return {"r": 0}

    @staticmethod
    def standard_prompt_wrap(x: str, y: str = "") -> str:
        return STANDARD_PROMPT.format(input=x) + y

    @staticmethod
    def cot_prompt_wrap(x: str, y: str = "") -> str:
        return COT_PROMPT.format(input=x) + y

    @staticmethod
    def propose_prompt_wrap(x: str, y: str = "") -> str:
        cur = _get_numbers(y if y else x)
        if cur == "24":
            return COT_PROMPT.format(input=x) + "Steps:" + y
        return PROPOSE_PROMPT.format(input=cur)

    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        last_line = y.strip().split("\n")[-1]
        if "left: " not in last_line:
            ans = last_line.lower().replace("answer: ", "")
            return VALUE_LAST_STEP_PROMPT.format(input=x, answer=ans)
        return VALUE_PROMPT.format(input=_get_numbers(y))

    @staticmethod
    def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
        if len(y.strip().split("\n")) == 4 and "answer" not in y.lower():
            return 0
        names = [v.split("\n")[-1] for v in value_outputs]
        value_map = {"impossible": 0.001, "likely": 1, "sure": 20}
        return sum(value * names.count(n) for n, value in value_map.items())


def _get_numbers(y: str) -> str:
    last = y.strip().split("\n")[-1]
    return last.split("left: ")[-1].split(")")[0]


# ============================================================================
# Search algorithm (Beam Search over LLM-generated branches)
# ============================================================================

def get_proposals(task, x, y, model, temperature):
    prompt = task.propose_prompt_wrap(x, y)
    proposals = gpt(prompt, model=model, temperature=temperature, n=1, stop=None)[0].split("\n")
    return [y + p + "\n" for p in proposals]


def get_samples(task, x, y, n_gen, prompt_sample, stop, model, temperature):
    if prompt_sample == "standard":
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == "cot":
        prompt = task.cot_prompt_wrap(x, y)
    else:
        raise ValueError(f"unknown prompt_sample: {prompt_sample}")
    samples = gpt(prompt, model=model, temperature=temperature, n=n_gen, stop=stop)
    return [y + s for s in samples]


def get_values(task, x, ys, n_eval, cache_value, model, temperature):
    values = []
    seen = {}
    for y in ys:
        if y in seen:
            values.append(0)
            continue
        vp = task.value_prompt_wrap(x, y)
        if cache_value and vp in task.value_cache:
            value = task.value_cache[vp]
        else:
            outputs = gpt(vp, model=model, temperature=temperature, n=n_eval, stop=None)
            value = task.value_outputs_unwrap(x, y, outputs)
            if cache_value:
                task.value_cache[vp] = value
        seen[y] = value
        values.append(value)
    return values


def get_votes(task, x, ys, n_eval, model, temperature):
    prompt = task.vote_prompt_wrap(x, ys)
    outputs = gpt(prompt, model=model, temperature=temperature, n=n_eval, stop=None)
    return task.vote_outputs_unwrap(outputs, len(ys))


def select(ys, values, method, n_select):
    ids = list(range(len(ys)))
    if method == "sample":
        ps = np.array(values) / sum(values)
        select_ids = np.random.choice(ids, size=min(n_select, len(ids)), p=ps).tolist()
    elif method == "greedy":
        select_ids = sorted(ids, key=lambda i: values[i], reverse=True)[:n_select]
    else:
        raise ValueError(f"unknown select: {method}")
    return [ys[i] for i in select_ids]


def solve(args, task, idx, to_print=True):
    """Core ToT loop: Generate → Evaluate → Select, repeated per step."""
    model = args.backend
    temperature = args.temperature
    x = task.get_input(idx)
    ys = [""]
    infos = []

    for step in range(task.steps):
        # ----- 1. Generate -----
        if args.method_generate == "sample":
            new_ys = [
                get_samples(task, x, y, args.n_generate_sample, args.prompt_sample,
                            task.stops[step], model, temperature)
                for y in ys
            ]
        elif args.method_generate == "propose":
            new_ys = [get_proposals(task, x, y, model, temperature) for y in ys]
        else:
            raise ValueError(f"unknown generate method: {args.method_generate}")
        new_ys = list(itertools.chain(*new_ys))
        if not new_ys:
            break

        # ----- 2. Evaluate -----
        if args.method_evaluate == "vote":
            values = get_votes(task, x, new_ys, args.n_evaluate_sample, model, temperature)
        elif args.method_evaluate == "value":
            values = get_values(task, x, new_ys, args.n_evaluate_sample, True, model, temperature)
        else:
            raise ValueError(f"unknown evaluate method: {args.method_evaluate}")

        # ----- 3. Select -----
        select_new_ys = select(new_ys, values, args.method_select, args.n_select_sample)

        if to_print:
            pairs = sorted(zip(new_ys, values), key=lambda p: p[1], reverse=True)
            sys.stdout.write(f"\n--- Step {step + 1} ---\n")
            for y, v in pairs[:5]:
                sys.stdout.write(f"  [{v:6.1f}] {y.strip()[-60:]}\n")
            sys.stdout.write(f"  -> kept: {[s.strip()[-40:] for s in select_new_ys]}\n")
            sys.stdout.flush()

        infos.append({
            "step": step, "x": x, "ys": ys,
            "new_ys": new_ys, "values": values, "select_new_ys": select_new_ys,
        })
        ys = select_new_ys

    return ys, {"steps": infos}


def naive_solve(args, task, idx, to_print=True):
    model = args.backend
    temperature = args.temperature
    x = task.get_input(idx)
    ys = get_samples(task, x, "", args.n_generate_sample, args.prompt_sample,
                     stop=None, model=model, temperature=temperature)
    return ys, {}


# ============================================================================
# Entry point
# ============================================================================

def parse_args():
    p = ArgumentParser(description="Tree of Thoughts — 24-Point Game Solver")

    p.add_argument("--backend", type=str, default=MODEL,
                   help="Model name or endpoint ID")
    p.add_argument("--temperature", type=float, default=0.7)

    p.add_argument("--start", type=int, default=8,
                   help="Start problem index (default: 8 = '4 5 6 10')")
    p.add_argument("--end", type=int, default=9,
                   help="End problem index (exclusive, default: 9)")

    p.add_argument("--naive", action="store_true",
                   help="Single-pass mode (no tree search)")
    p.add_argument("--prompt_sample", type=str, choices=["standard", "cot"],
                   default="standard")

    p.add_argument("--method_generate", type=str,
                   choices=["sample", "propose"], default="propose")
    p.add_argument("--method_evaluate", type=str,
                   choices=["value", "vote"], default="value")
    p.add_argument("--method_select", type=str,
                   choices=["sample", "greedy"], default="greedy")
    p.add_argument("--n_generate_sample", type=int, default=1)
    p.add_argument("--n_evaluate_sample", type=int, default=1)
    p.add_argument("--n_select_sample", type=int, default=1)

    return p.parse_args()


def main():
    args = parse_args()
    print(args)
    task = Game24Task()

    n_correct = 0
    n_total = args.end - args.start

    for i in range(args.start, args.end):
        print(f"\n{'=' * 50}")
        print(f"Problem {i}: {task.get_input(i)}")
        print(f"{'=' * 50}")

        if args.naive:
            ys, info = naive_solve(args, task, i)
        else:
            ys, info = solve(args, task, i)

        # Test each candidate answer
        for j, y in enumerate(ys):
            result = task.test_output(i, y)
            status = "CORRECT" if result["r"] else "WRONG"
            n_correct += result["r"]
            print(f"\n  Candidate {j + 1}: {status}")
            print(f"  {y.strip()}")

    print(f"\n{'=' * 50}")
    print(f"Accuracy: {n_correct}/{n_total} = {n_correct / n_total:.1%}")
    print(f"Usage: {json.dumps(gpt_usage(), indent=2)}")


if __name__ == "__main__":
    main()
