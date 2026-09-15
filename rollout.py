# -*- coding: utf-8 -*-
"""轨迹采样：让当前模型对每道题跑 N 次，用规则奖励打分。

这是 Agentic RL 的"采样"环节。用较高温度获得多样性——因为我们要的就是
「同一道题有的成功、有的失败」，才能做拒绝采样与偏好对。
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from env import TOOLS, parse_answer, parse_call, reward, run_tool

SYS = (
    "你是客服工具调用助手。看到用户问题后：\n"
    "第一步：只输出一次工具调用，格式严格为 <tool_call>{\"name\": \"工具名\", \"args\": {...}}</tool_call>\n"
    "可用工具：" + json.dumps(TOOLS, ensure_ascii=False) + "\n"
    "第二步：看到工具返回结果后，用 <answer>...</answer> 给出简洁回答，如实转述工具结果。"
)
FEWSHOT = ("用户：帮我查一下订单 SO20260810001 的物流\n"
           "第一步：<tool_call>{\"name\": \"track_logistics\", \"args\": {\"order_id\": \"SO20260810001\"}}</tool_call>")


def load(path: str):
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = dict(device_map="auto", trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, **kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, **kw)
    model.eval()
    return model, tok


def chat(tok, system: str, user: str) -> str:
    return tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def gen(model, tok, prompt: str, max_new_tokens: int, temperature: float) -> str:
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=max_new_tokens,
                         do_sample=temperature > 0, temperature=max(temperature, 1e-5),
                         top_p=0.95, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="轨迹采样 + 规则奖励")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", default="data/tasks_train.jsonl")
    ap.add_argument("--out", default="data/rollouts.jsonl")
    ap.add_argument("--n", type=int, default=4, help="每题采样条数")
    ap.add_argument("--limit", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    a = ap.parse_args()

    tasks = [json.loads(l) for l in open(a.tasks, encoding="utf-8")][: a.limit]
    model, tok = load(a.model)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    fout = open(a.out, "w", encoding="utf-8")
    stats = {"total": 0, "success": 0}
    for i, t in enumerate(tasks, 1):
        for k in range(a.n):
            # ---- 第 1 步：模型给工具调用 ----
            p1 = chat(tok, SYS, FEWSHOT + "\n\n用户：" + t["question"] + "\n第一步：")
            step1 = gen(model, tok, p1, a.max_new_tokens, a.temperature)
            name, args = parse_call(step1)
            # ---- 环境执行 ----
            obs = run_tool(name, args) if name else {"error": "no tool call"}
            # ---- 第 2 步：模型据结果作答 ----
            p2 = chat(tok, SYS, FEWSHOT + "\n\n用户：" + t["question"]
                      + "\n第一步：" + step1 + "\n工具返回：" + json.dumps(obs, ensure_ascii=False) + "\n第二步：")
            step2 = gen(model, tok, p2, a.max_new_tokens, a.temperature)

            score, detail = reward(step1, step2, t["gold_tool"], t["gold_args"], t["gold_entities"])
            stats["total"] += 1
            stats["success"] += int(score >= 0.999)
            fout.write(json.dumps({
                "task_id": t["task_id"], "question": t["question"],
                "gold_tool": t["gold_tool"], "gold_args": t["gold_args"],
                "gold_entities": t["gold_entities"], "meta": t["meta"],
                "prompt1": p1, "prompt2": p2,
                "step1": step1, "observation": obs, "step2": step2,
                "score": score, "detail": detail,
            }, ensure_ascii=False) + "\n")
        if i % 20 == 0:
            print(f"  {i}/{len(tasks)} 题，累计成功率 {stats['success']/stats['total']:.1%}")
    fout.close()
    print(f"\n采样完成：{stats['total']} 条轨迹，成功（满分）{stats['success']} 条，"
          f"成功率 {stats['success']/stats['total']:.1%}")
    print(f"写入 {a.out}")


if __name__ == "__main__":
    main()
