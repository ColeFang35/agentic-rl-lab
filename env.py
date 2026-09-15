# -*- coding: utf-8 -*-
"""可验证的工具环境 + 规则奖励（Agentic RL 的"环境"与"奖励"）。

为什么不用模型当裁判：
Agentic RL 最怕 **reward hacking**——模型学会讨好裁判而不是解决问题。
所以这里的奖励全部来自**可验证信号**：工具名对不对、参数对不对、最终回答有没有命中真实实体。
"""
from __future__ import annotations

import json
import re

# ---------------- 工具定义（给模型看的目录） ----------------
TOOLS = [
    {"name": "track_logistics", "desc": "查询订单物流轨迹", "args": {"order_id": "订单号"}},
    {"name": "query_order", "desc": "查询订单详情与状态", "args": {"order_id": "订单号"}},
    {"name": "check_refund_policy", "desc": "核查订单是否可退款", "args": {"order_id": "订单号"}},
    {"name": "apply_refund", "desc": "提交退款申请", "args": {"order_id": "订单号", "reason": "原因"}},
    {"name": "search_faq", "desc": "检索退换货/运费等政策", "args": {"question": "问题"}},
]

# ---------------- 假数据（让"环境"能返回真实结构的结果） ----------------
_ORDERS = {
    "SO20260810001": {"status": "已发货", "carrier": "云雀速运", "latest": "派送中，快递员赵师傅"},
    "SO20260812003": {"status": "待发货", "carrier": "—", "latest": "商家尚未发货"},
    "SO20260903011": {"status": "已签收", "carrier": "顺丰速运", "latest": "已签收，签收人：本人"},
    "SO20260101001": {"status": "已签收", "carrier": "京东物流", "latest": "已签收"},
}
_FAQ = {
    "退货": "支持 7 天无理由退货，商品需不影响二次销售。",
    "运费": "单笔满 99 元包邮，偏远地区另计。",
    "发票": "支持开具电子发票，下单后可申请。",
}

# 传给 apply_chat_template 的工具 schema（OpenAI 风格）——让模型用它"训练时的格式"输出
TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": t["name"], "description": t["desc"],
        "parameters": {"type": "object",
                       "properties": {k: {"type": "string", "description": v}
                                      for k, v in t["args"].items()},
                       "required": list(t["args"].keys())}}}
    for t in TOOLS
]

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S)


def run_tool(name: str, args: dict) -> dict:
    """执行工具（模拟业务查询）。"""
    oid = str(args.get("order_id", ""))
    if name in ("track_logistics", "query_order"):
        o = _ORDERS.get(oid)
        if not o:
            return {"found": False, "message": "未查询到该订单"}
        return {"found": True, "order_id": oid, "status": o["status"],
                "carrier": o["carrier"], "latest": o["latest"]}
    if name == "check_refund_policy":
        o = _ORDERS.get(oid)
        if not o:
            return {"found": False, "message": "未查询到该订单"}
        ok = o["status"] in ("待发货", "已发货")
        return {"found": True, "order_id": oid, "refundable": ok,
                "reason": "" if ok else "已签收订单需走售后流程"}
    if name == "apply_refund":
        return {"success": True, "order_id": oid, "ticket_id": "TK" + oid[-4:]}
    if name == "search_faq":
        q = str(args.get("question", ""))
        for k, v in _FAQ.items():
            if k in q:
                return {"found": True, "answer": v}
        return {"found": False, "message": "知识库未收录"}
    return {"error": f"未知工具 {name}"}


# ---------------- 可验证奖励 ----------------
def parse_call(text: str) -> tuple[str | None, dict]:
    m = TOOL_CALL_RE.search(text or "")
    if not m:
        return None, {}
    try:
        d = json.loads(m.group(1))
        return d.get("name"), (d.get("arguments") or d.get("args") or {})
    except ValueError:
        return None, {}


def parse_answer(text: str) -> str:
    m = ANSWER_RE.search(text or "")
    return (m.group(1).strip() if m else "")


def reward(step1_text: str, step2_text: str, gold_name: str,
           gold_args: dict, gold_entities: list[str]) -> tuple[float, dict]:
    """规则奖励：工具名 0.4 + 参数 0.3 + 最终答案命中实体 0.3。"""
    name, args = parse_call(step1_text)
    detail = {}

    r_name = 0.4 if name == gold_name else 0.0
    detail["tool_name"] = r_name

    r_args = 0.0
    if name == gold_name and args:
        hit = sum(1 for k, v in gold_args.items() if str(args.get(k)) == str(v))
        r_args = 0.3 * hit / len(gold_args) if gold_args else 0.0
    detail["args"] = round(r_args, 3)

    ans = parse_answer(step2_text)
    hit_ent = [e for e in gold_entities if e in ans]
    r_ans = 0.3 * (len(hit_ent) / len(gold_entities)) if gold_entities else 0.0
    detail["answer"] = round(r_ans, 3)

    total = round(r_name + r_args + r_ans, 3)
    return total, detail
