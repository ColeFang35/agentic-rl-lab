# -*- coding: utf-8 -*-
"""对比评测：基座 / RFT / DPO 三种模型在留出集上的**成功率**与平均奖励。

用**贪心解码**（可复现），同一套测试任务，只换模型 —— 直接回答"RL 到底有没有用"。
"""
from __future__ import annotations

import argparse
import json
import os
from statistics import mean

import torch
from peft import PeftModel

from env import parse_call, reward, run_tool
from rollout import FEWSHOT, SYS, chat, gen, load


def rollout_once(model, tok, task: dict, max_new_tokens: int) -> dict:
    p1 = chat(tok, SYS, FEWSHOT + "\n\n用户：" + task["question"] + "\n第一步：")
    step1 = gen(model, tok, p1, max_new_tokens, temperature=0.0)     # 贪心
    name, args = parse_call(step1)
    obs = run_tool(name, args) if name else {"error": "no tool call"}
    p2 = chat(tok, SYS, FEWSHOT + "\n\n用户：" + task["question"]
              + "\n第一步：" + step1 + "\n工具返回：" + json.dumps(obs, ensure_ascii=False) + "\n第二步：")
    step2 = gen(model, tok, p2, max_new_tokens, temperature=0.0)
    score, detail = reward(step1, step2, task["gold_tool"], task["gold_args"], task["gold_entities"])
    return {"question": task["question"], "step1": step1, "step2": step2,
            "score": score, "detail": detail}


def eval_model(model, tok, tasks: list[dict], max_new_tokens: int) -> tuple[dict, list[dict]]:
    rows = [rollout_once(model, tok, t, max_new_tokens) for t in tasks]
    n = len(rows)
    m = {
        "success_rate": round(sum(1 for r in rows if r["score"] >= 0.999) / n, 3),
        "mean_reward": round(mean(r["score"] for r in rows), 3),
        "tool_name_acc": round(mean(r["detail"]["tool_name"] / 0.4 for r in rows), 3),
    }
    return m, rows


def main() -> None:
    ap = argparse.ArgumentParser(description="基座 vs RFT vs DPO 对比")
    ap.add_argument("--base", required=True)
    ap.add_argument("--rft", default="outputs/rft-adapter")
    ap.add_argument("--dpo", default="outputs/dpo-adapter")
    ap.add_argument("--tasks", default="data/tasks_test.jsonl")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--out", default="reports/rl_compare.json")
    a = ap.parse_args()

    tasks = [json.loads(l) for l in open(a.tasks, encoding="utf-8")][: a.limit]
    print(f"测试任务：{len(tasks)} 条\n")

    results, samples = {}, {}
    for tag, adapter in (("base", None), ("rft", a.rft), ("dpo", a.dpo)):
        if adapter and not os.path.isdir(adapter):
            print(f"[{tag}] 适配器不存在，跳过（{adapter}）")
            continue
        model, tok = load(a.base)
        if adapter:
            model = PeftModel.from_pretrained(model, adapter)
            model.eval()
        m, rows = eval_model(model, tok, tasks, a.max_new_tokens)
        results[tag] = m
        samples[tag] = rows[:3]
        print(f"[{tag:>4}] 成功率 {m['success_rate']:.1%} | 平均奖励 {m['mean_reward']:.3f} "
              f"| 工具名正确率 {m['tool_name_acc']:.1%}")
        del model
        torch.cuda.empty_cache()

    if "base" in results:
        for tag in ("rft", "dpo"):
            if tag in results:
                results[f"{tag}_delta"] = {
                    k: round(results[tag][k] - results["base"][k], 3)
                    for k in ("success_rate", "mean_reward", "tool_name_acc")}
                print(f"\n[{tag} 相对基座] {results[f'{tag}_delta']}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"metrics": results, "samples": samples}, f, ensure_ascii=False, indent=2)
    print(f"\n结果写入：{a.out}")

    if "base" in samples:
        print("\n=== 样例（基座 vs 最优档）===")
        best = "dpo" if "dpo" in samples else ("rft" if "rft" in samples else "base")
        for b, o in zip(samples["base"], samples[best]):
            print(f"\n问：{b['question']}")
            print(f"  基座  ({b['score']:.2f})：{b['step1'][:70]}")
            print(f"  {best}  ({o['score']:.2f})：{o['step1'][:70]}")


if __name__ == "__main__":
    main()
