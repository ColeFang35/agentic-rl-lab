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

from env import TOOL_SCHEMAS, parse_answer, parse_call, reward, run_tool

SYS = "你是云雀商城的客服助手。请使用提供的工具查询真实数据后再回答，不要凭空编造。"

# 用 Qwen 原生的工具调用模板：把 tools 交给 apply_chat_template，
# 模型就会用它在训练时见过的 <tool_call> 格式（注意是 arguments 不是 args）
def build_prompt_step1(tok, question: str) -> str:
    return tok.apply_chat_template(
        [{"role": "system", "content": SYS}, {"role": "user", "content": question}],
        tools=TOOL_SCHEMAS, tokenize=False, add_generation_prompt=True)


def build_prompt_step2(tok, question: str, step1: str, observation: dict, tool_name: str) -> str:
    return tok.apply_chat_template(
        [{"role": "system", "content": SYS},
         {"role": "user", "content": question},
         {"role": "assistant", "content": step1},
         {"role": "tool", "name": tool_name, "content": json.dumps(observation, ensure_ascii=False)}],
        tools=TOOL_SCHEMAS, tokenize=False, add_generation_prompt=True)


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
            p1 = build_prompt_step1(tok, t["question"])
            step1 = gen(model, tok, p1, a.max_new_tokens, a.temperature)
            name, args = parse_call(step1)
            # ---- 环境执行 ----
            obs = run_tool(name, args) if name else {"error": "no tool call"}
            # ---- 第 2 步：模型据结果作答 ----
            p2 = build_prompt_step2(tok, t["question"], step1, obs, name or "unknown")
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
