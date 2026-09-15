# -*- coding: utf-8 -*-
"""任务集：用户问题 + 标准工具调用 + 期望实体（全可验证）。"""
from __future__ import annotations

import argparse
import json
import os
import random

random.seed(20260915)

ORDERS = [("SO20260810001", "已发货", "云雀速运"),
          ("SO20260812003", "待发货", "—"),
          ("SO20260903011", "已签收", "顺丰速运"),
          ("SO20260101001", "已签收", "京东物流")]

# (意图, 问题模板, 工具, 参数构造, 期望实体构造)
SPECS = [
    ("查物流", ["帮我查一下订单 {o} 的物流", "看看 {o} 到哪了", "{o} 的快递什么情况"],
     "track_logistics", lambda o, s, c: {"order_id": o}, lambda o, s, c: [o, s]),
    ("查订单", ["订单 {o} 现在什么状态", "帮我查下 {o} 这个单子", "{o} 是什么情况"],
     "query_order", lambda o, s, c: {"order_id": o}, lambda o, s, c: [o, s]),
    ("退款咨询", ["{o} 能退款吗", "我想退 {o}", "{o} 支持退货吗"],
     "check_refund_policy", lambda o, s, c: {"order_id": o}, lambda o, s, c: [o]),
    ("申请退款", ["帮我退掉 {o}", "我要申请退 {o} 这个订单"],
     "apply_refund", lambda o, s, c: {"order_id": o, "reason": "不想要了"}, lambda o, s, c: [o]),
    ("政策问答", ["退货规则是什么", "运费怎么算", "发票怎么开"],
     "search_faq", lambda o, s, c: {"question": "退货规则"}, lambda o, s, c: []),
]


def gen(n: int) -> list[dict]:
    out = []
    for _ in range(n):
        intent, qs, tool, mk_args, mk_ents = random.choice(SPECS)
        o, s, c = random.choice(ORDERS)
        q = random.choice(qs).format(o=o)
        args = mk_args(o, s, c)
        if intent == "政策问答":
            args = {"question": q}
        out.append({
            "task_id": f"t{len(out):04d}",
            "question": q,
            "gold_tool": tool,
            "gold_args": args,
            "gold_entities": mk_ents(o, s, c),
            "meta": {"intent": intent, "order": o},
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=160)
    ap.add_argument("--test", type=int, default=40)
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)) + "/data")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for name, n in (("tasks_train.jsonl", a.train), ("tasks_test.jsonl", a.test)):
        rows = gen(n)
        with open(os.path.join(a.out, name), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"写入 {name}: {len(rows)} 条")
    from collections import Counter
    print("意图分布:", dict(Counter(r["meta"]["intent"] for r in gen(a.train)).most_common()))


if __name__ == "__main__":
    main()
