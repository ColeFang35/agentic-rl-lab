# -*- coding: utf-8 -*-
"""把采样轨迹加工成两种训练数据：

1) **RFT（拒绝采样 + SFT）**：只保留满分轨迹 → 直接做指令微调数据
   —— 这是最简单的 agentic RL：不写 RL 算法，只用"成功了再学一遍"提升成功率

2) **DPO 偏好对**：同一道题里，得分最高的当 chosen、得分最低的当 rejected
   —— 偏好对**自动生成**，不需要人工标注（这是 agentic 场景的常见做法）
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

from env import parse_call


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", default="data/rollouts.jsonl")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--pass-score", type=float, default=0.999, help="视为满分/成功的分数阈值")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.rollouts, encoding="utf-8")]
    print(f"读入轨迹 {len(rows)} 条")

    # ---------- 1) RFT：只留满分轨迹 ----------
    sft: list[dict] = []
    for r in rows:
        if r["score"] < a.pass_score:
            continue
        # 两个阶段各作为一条训练样本：学"该怎么调工具"和"该怎么答"
        sft.append({"stage": "tool_call", "prompt": r["prompt1"], "completion": r["step1"],
                    "meta": r["meta"]})
        if "<answer>" in r["step2"]:
            sft.append({"stage": "answer", "prompt": r["prompt2"], "completion": r["step2"],
                        "meta": r["meta"]})

    # ---------- 2) DPO：同题最优/最差配对 ----------
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_task[r["task_id"]].append(r)
    pairs: list[dict] = []
    for tid, group in by_task.items():
        if len(group) < 2:
            continue
        best = max(group, key=lambda x: x["score"])
        worst = min(group, key=lambda x: x["score"])
        if best["score"] <= worst["score"]:
            continue          # 全一样（都成功或都失败）→ 没有偏好信号
        # 只在"工具调用"这个决策点上做偏好（agentic 的关键决策）
        if parse_call(best["step1"])[0] == parse_call(worst["step1"])[0] and best["score"] == worst["score"]:
            continue
        pairs.append({"prompt": best["prompt1"], "chosen": best["step1"],
                      "rejected": worst["step1"], "meta": best["meta"],
                      "score_gap": round(best["score"] - worst["score"], 3)})

    os.makedirs(a.out_dir, exist_ok=True)
    for name, data in (("sft.jsonl", sft), ("dpo_pairs.jsonl", pairs)):
        path = os.path.join(a.out_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"写入 {name}: {len(data)} 条")

    n_tasks = len(by_task)
    ok_tasks = sum(1 for g in by_task.values() if any(x["score"] >= a.pass_score for x in g))
    print(f"\n任务数 {n_tasks}，其中至少有一条成功轨迹的：{ok_tasks}（{ok_tasks/n_tasks:.0%}）")
    print("→ 这些成功轨迹就是 RFT 的种子数据；失败与成功并存的任务构成 DPO 偏好对。")


if __name__ == "__main__":
    main()
